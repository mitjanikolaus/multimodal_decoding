import argparse
import time

import numpy as np
import sklearn
import os
import pickle

import torch

from data import LatentFeatsConfig, SELECT_DEFAULT, FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, \
    LANG_FEAT_COMBINATION_CHOICES, apply_mask, standardize_fmri_betas, get_latent_features, \
    standardize_latents, MODALITY_AGNOSTIC, TRAINING_MODES, SPLIT_TRAIN, SPLIT_IMAGERY, get_fmri_data, \
    ALL_SPLITS, TEST_SPLITS, SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS
from eval import pairwise_accuracy, calc_all_pairwise_accuracy_scores, ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, \
    ACC_IMAGERY_WHOLE_TEST
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from utils import FMRI_BETAS_DIR, SUBJECTS, RESULTS_FILE, DEFAULT_MODEL, DEFAULT_RESOLUTION, \
    ATTENTION_MOD_FMRI_BETAS_DIR, RIDGE_DECODER_ATTN_MOD_OUT_DIR

NUM_CV_SPLITS = 5
DEFAULT_ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


def get_run_str(betas_dir, feats_config, mask=None, surface=False, resolution=DEFAULT_RESOLUTION):
    run_str = f"{feats_config.model}_{feats_config.combined_feats}"
    run_str += f"_{feats_config.vision_features}"
    run_str += f"_{feats_config.lang_features}"
    if betas_dir.endswith(os.sep):
        betas_dir = betas_dir[:-1]
    run_str += f"_{betas_dir.split(os.sep)[-1]}"

    if mask is not None:
        if mask.startswith("functional_") or mask.startswith("anatomical_"):
            run_str += f"_mask_{mask}"
        else:
            mask_name = os.path.basename(mask).replace(".p", "")
            run_str += f"_mask_{mask_name}"

    if surface:
        run_str += f"_surface_{resolution}"

    return run_str


def tensor_pairwise_accuracy(
        latents, predictions, metric="cosine", standardize_predictions=False, standardize_latents=False
):
    if torch.is_tensor(latents) and latents.is_cuda:
        latents = latents.cpu().numpy()
        predictions = predictions.cpu().numpy()
    predictions = predictions.squeeze()

    return pairwise_accuracy(latents, predictions, metric, standardize_predictions, standardize_latents)


def get_fmri_data_for_splits(subject, splits, training_mode, betas_dir, attn_mod_betas_dir=None, surface=False,
                             resolution=DEFAULT_RESOLUTION):
    fmri_betas, stim_ids, stim_types = dict(), dict(), dict()
    for split in splits:
        mode = training_mode if split == SPLIT_TRAIN else MODALITY_AGNOSTIC
        betas_dir = betas_dir if split in [SPLIT_TRAIN, SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS, SPLIT_IMAGERY] else attn_mod_betas_dir
        fmri_betas[split], stim_ids[split], stim_types[split] = get_fmri_data(
            betas_dir,
            subject,
            split,
            mode,
            surface=surface,
            resolution=resolution,
        )

    return fmri_betas, stim_ids, stim_types


def get_latents_for_splits(subject, feats_config, splits, training_mode):
    latents = dict()
    for split in splits:
        mode = training_mode if split == SPLIT_TRAIN else MODALITY_AGNOSTIC
        latents[split] = get_latent_features(feats_config, subject, split, mode)
    return latents


def run(args):
    if torch.cuda.is_available() and not args.force_cpu:
        print("Setting backend to cuda")
        backend = set_backend("torch_cuda")
    else:
        backend = set_backend("numpy")

    for training_mode in args.training_modes:
        for subject in args.subjects:
            fmri_betas_full, stim_ids, stim_types = get_fmri_data_for_splits(
                subject, ALL_SPLITS, training_mode, args.betas_dir, args.attn_mod_betas_dir, args.surface,
                args.resolution
            )
            for mask in args.masks:
                mask = None if mask in ["none", "None"] else mask
                fmri_betas = apply_mask(mask, fmri_betas_full, args)
                fmri_betas = standardize_fmri_betas(fmri_betas)
                print('\n\n')
                for split in fmri_betas.keys():
                    print(f"{split} fMRI betas shape: {fmri_betas[split].shape}")

                for model in args.models:
                    feats_config = LatentFeatsConfig(
                        model, args.features, args.test_features, args.vision_features, args.lang_features
                    )

                    print(f"TRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                          f"MODEL: {model} | FEATURES: {feats_config.features} {feats_config.vision_features} "
                          f"{feats_config.lang_features} | TEST FEATURES: {feats_config.test_features}")
                    if mask is not None:
                        print("Mask: ", os.path.basename(mask))

                    run_str = get_run_str(args.betas_dir, feats_config, mask, args.surface, args.resolution)
                    results_file_path = os.path.join(
                        RIDGE_DECODER_ATTN_MOD_OUT_DIR, training_mode, subject, run_str, RESULTS_FILE
                    )
                    if os.path.isfile(results_file_path) and not args.overwrite:
                        print(f"Skipping decoder training as results are already present at {results_file_path}")
                        continue

                    latents = get_latents_for_splits(subject, feats_config, ALL_SPLITS, training_mode)
                    latents = standardize_latents(latents)
                    print(f"train latents shape: {latents[SPLIT_TRAIN].shape}")

                    # skip input data checking to limit memory use
                    # (https://gallantlab.org/himalaya/troubleshooting.html?highlight=cuda)
                    sklearn.set_config(assume_finite=True)

                    clf = KernelRidgeCV(
                        cv=NUM_CV_SPLITS,
                        alphas=args.l2_regularization_alphas,
                        solver_params=dict(
                            n_targets_batch=args.n_targets_batch,
                            n_alphas_batch=args.n_alphas_batch,
                            n_targets_batch_refit=args.n_targets_batch_refit,
                            score_func=tensor_pairwise_accuracy,
                            local_alpha=False,
                        )
                    )

                    start = time.time()
                    clf.fit(fmri_betas[SPLIT_TRAIN], latents[SPLIT_TRAIN])
                    end = time.time()
                    print(f"Elapsed time: {int(end - start)}s")

                    best_alpha = np.round(backend.to_numpy(clf.best_alphas_[0]))

                    predicted_latents = {split: clf.predict(fmri_betas[split]) for split in TEST_SPLITS}

                    predicted_latents = {split: backend.to_numpy(lats) for split, lats in predicted_latents.items()}

                    results = {
                        "alpha": best_alpha,
                        "model": model,
                        "subject": subject,
                        "features": feats_config.features,
                        "test_features": feats_config.test_features,
                        "vision_features": feats_config.vision_features,
                        "lang_features": feats_config.lang_features,
                        "training_mode": training_mode,
                        "mask": mask,
                        "num_voxels": fmri_betas[SPLIT_TRAIN].shape[1],
                        "predictions": predicted_latents,
                        "surface": args.surface,
                        "resolution": args.resolution,
                    }
                    scores = calc_all_pairwise_accuracy_scores(
                        latents, predicted_latents, stim_types, standardize_predictions=True
                    )
                    results.update(scores)
                    print(
                        f"Best alphas: {best_alpha}"
                    )

                    # results_no_standardization = calc_all_pairwise_accuracy_scores(
                    #     latents, predicted_latents, stim_types, standardize_predictions=False
                    # )
                    # print(
                    #     f"Without standardization of predictions:\n"
                    #     f"Pairwise acc (captions): {results[ACC_CAPTIONS]:.2f}"
                    #     f" | Pairwise acc (images): {results[ACC_IMAGES]:.2f}"
                    #     f" | Pairwise acc (imagery): {results[ACC_IMAGERY]:.2f}"
                    #     f" | Pairwise acc (imagery whole test set): "
                    #     f"{results[ACC_IMAGERY_WHOLE_TEST]:.2f}"
                    # )
                    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                    pickle.dump(results, open(results_file_path, 'wb'))

                    print(scores)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)
    parser.add_argument("--attn-mod-betas-dir", type=str, default=ATTENTION_MOD_FMRI_BETAS_DIR)

    parser.add_argument("--training-modes", type=str, nargs="+", default=[MODALITY_AGNOSTIC],
                        choices=TRAINING_MODES)

    parser.add_argument("--models", type=str, nargs='+', default=[DEFAULT_MODEL])
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--vision-features", type=str, default=SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+', default=DEFAULT_ALPHAS)

    parser.add_argument("--n-targets-batch", type=int, default=1024)
    parser.add_argument("--n-targets-batch-refit", type=int, default=1024)
    parser.add_argument("--n-alphas-batch", type=int, default=1)

    parser.add_argument("--surface", action='store_true', default=False)
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--masks", nargs='+', type=str, default=[None])

    parser.add_argument("--overwrite", action='store_true', default=False)

    parser.add_argument("--force-cpu", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args)
