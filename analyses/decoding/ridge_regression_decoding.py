import argparse
import time
from collections import Counter

import numpy as np
import sklearn
import os
import pickle

from data import LatentFeatsConfig, SELECT_DEFAULT, FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, \
    LANG_FEAT_COMBINATION_CHOICES, get_fmri_data, apply_mask, standardize_fmri_betas, get_latent_features, \
    standardize_latents, TESTING_MODE, IMAGERY, remove_nans
from eval import pairwise_accuracy, calc_all_pairwise_accuracy_scores, ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, \
    ACC_IMAGERY_WHOLE_TEST
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from utils import FMRI_BETAS_DIR, SUBJECTS, DEFAULT_RESOLUTION, RESULTS_FILE, MODE_AGNOSTIC, TRAIN_MODE_CHOICES, \
    RIDGE_DECODER_OUT_DIR

NUM_CV_SPLITS = 5
DEFAULT_ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


def get_run_str(betas_dir, feats_config, mask=None, surface=False, resolution=None, hemi=None):
    run_str = f"{feats_config.model}_{feats_config.combined_feats}"
    run_str += f"_{feats_config.vision_features}"
    run_str += f"_{feats_config.lang_features}"
    if betas_dir.endswith(os.sep):
        betas_dir = betas_dir[:-1]
    run_str += f"_{betas_dir.split(os.sep)[-1]}"

    if mask is not None:
        if mask.startswith("functional_") or mask.startswith("anatomical_"):
            run_str += f"_mask_{mask}"
        elif "p_values" in mask:
            mask_name = os.path.basename(mask).replace(".p", "")
            run_str += f"_mask_{mask_name}"
        else:
            raise RuntimeError(f"Unsupported mask: {mask}")

    if surface:
        run_str += f"_surface_{resolution}"

    if hemi:
        run_str += f"_{hemi}_hemi"

    return run_str


def tensor_pairwise_accuracy(
        latents, predictions, metric="cosine", standardize_predictions=False, standardize_latents=False
):
    latents = latents.cpu().numpy()
    predictions = predictions.cpu().numpy().squeeze()

    return pairwise_accuracy(latents, predictions, metric, standardize_predictions, standardize_latents)


def run(args):
    if args.cuda:
        print("Setting backend to cuda")
        backend = set_backend("torch_cuda")
    else:
        backend = set_backend("numpy")

    for training_mode in args.training_modes:
        for subject in args.subjects:
            train_fmri_betas_full, train_stim_ids, train_stim_types = get_fmri_data(
                args.betas_dir,
                subject,
                training_mode,
                surface=args.surface,
                resolution=args.resolution,
            )
            test_fmri_betas_full, test_stim_ids, test_stim_types = get_fmri_data(
                args.betas_dir,
                subject,
                TESTING_MODE,
                surface=args.surface,
                resolution=args.resolution,
            )
            imagery_fmri_betas_full, imagery_stim_ids, imagery_stim_types = get_fmri_data(
                args.betas_dir,
                subject,
                IMAGERY,
                surface=args.surface,
                resolution=args.resolution,
            )
            for mask in args.masks:
                mask = None if mask in ["none", "None"] else mask
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = apply_mask(
                    mask, [train_fmri_betas_full, test_fmri_betas_full, imagery_fmri_betas_full], args
                )
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = remove_nans(
                    [train_fmri_betas, test_fmri_betas, imagery_fmri_betas]
                )
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = standardize_fmri_betas(
                    train_fmri_betas, test_fmri_betas, imagery_fmri_betas,
                )

                for model in args.models:
                    feats_config = LatentFeatsConfig(
                        model, args.features, args.test_features, args.vision_features, args.lang_features
                    )

                    train_latents = get_latent_features(feats_config, train_stim_ids, train_stim_types)
                    test_latents = get_latent_features(
                        feats_config, test_stim_ids, test_stim_types, test_mode=True
                    )
                    imagery_latents = get_latent_features(
                        feats_config, imagery_stim_ids, imagery_stim_types, test_mode=True
                    )
                    train_latents, test_latents, imagery_latents = standardize_latents(
                        train_latents, test_latents, imagery_latents
                    )

                    print(f"\nTRAIN MODE: {training_mode} | MASK: {mask} | SUBJECT: {subject} | "
                          f"MODEL: {model} | FEATURES: {feats_config.features} {feats_config.vision_features} "
                          f"{feats_config.lang_features} | TEST FEATURES: {feats_config.test_features}")
                    print(f"train fMRI betas shape: {train_fmri_betas.shape}")
                    print(f"test fMRI betas shape: {test_fmri_betas.shape}")
                    print(f"imagery fMRI betas shape: {imagery_fmri_betas.shape}")
                    print(f"train latents shape: {train_latents.shape}")

                    run_str = get_run_str(args.betas_dir, feats_config, mask, args.surface, args.resolution)
                    results_file_path = os.path.join(
                        RIDGE_DECODER_OUT_DIR, training_mode, subject, run_str, RESULTS_FILE
                    )
                    if os.path.isfile(results_file_path) and not args.overwrite:
                        print(f"Skipping decoder training as results are already present at {results_file_path}")
                        continue

                    clf = KernelRidgeCV(
                        cv=NUM_CV_SPLITS,
                        alphas=args.l2_regularization_alphas,
                        solver_params=dict(
                            n_targets_batch=args.n_targets_batch,
                            n_alphas_batch=args.n_alphas_batch,
                            n_targets_batch_refit=args.n_targets_batch_refit,
                            score_func=tensor_pairwise_accuracy,
                        )
                    )

                    train_latents = train_latents.astype(np.float32)
                    train_fmri_betas = train_fmri_betas.astype(np.float32)

                    # skip input data checking to limit memory use
                    # (https://gallantlab.org/himalaya/troubleshooting.html?highlight=cuda)
                    sklearn.set_config(assume_finite=True)

                    start = time.time()
                    clf.fit(train_fmri_betas, train_latents)
                    end = time.time()
                    print(f"Elapsed time: {int(end - start)}s")

                    best_alphas = np.round(backend.to_numpy(clf.best_alphas_))

                    test_fmri_betas = test_fmri_betas.astype(np.float32)
                    test_predicted_latents = clf.predict(test_fmri_betas)

                    imagery_fmri_betas = imagery_fmri_betas.astype(np.float32)
                    imagery_predicted_latents = clf.predict(imagery_fmri_betas)

                    imagery_predicted_latents = backend.to_numpy(imagery_predicted_latents)
                    test_predicted_latents = backend.to_numpy(test_predicted_latents)

                    results = {
                        "alphas": best_alphas,
                        "model": model,
                        "subject": subject,
                        "features": feats_config.features,
                        "test_features": feats_config.test_features,
                        "vision_features": feats_config.vision_features,
                        "lang_features": feats_config.lang_features,
                        "training_mode": training_mode,
                        "mask": mask,
                        "num_voxels": test_fmri_betas.shape[1],
                        "stimulus_ids": test_stim_ids,
                        "stimulus_types": test_stim_types,
                        "imagery_stimulus_ids": imagery_stim_ids,
                        "predictions": test_predicted_latents,
                        "imagery_predictions": imagery_predicted_latents,
                        "latents": test_latents,
                        "imagery_latents": imagery_latents,
                        "surface": args.surface,
                        "resolution": args.resolution,
                    }
                    scores = calc_all_pairwise_accuracy_scores(
                        test_latents, test_predicted_latents, test_stim_types,
                        imagery_latents, imagery_predicted_latents, standardize_predictions=True
                    )
                    results.update(scores)
                    print(
                        f"Best alphas: {Counter(best_alphas)}\n"
                        f"Pairwise acc (captions): {results[ACC_CAPTIONS]:.2f}"
                        f" | Pairwise acc (images): {results[ACC_IMAGES]:.2f}"
                        f" | Pairwise acc (imagery): {results[ACC_IMAGERY]:.2f}"
                        f" | Pairwise acc (imagery whole test set): {results[ACC_IMAGERY_WHOLE_TEST]:.2f}"
                    )

                    results_no_standardization = calc_all_pairwise_accuracy_scores(
                        test_latents, test_predicted_latents, test_stim_types,
                        imagery_latents, imagery_predicted_latents, standardize_predictions=False
                    )
                    print(
                        f"Without standardization of predictions:\n"
                        f"Pairwise acc (captions): {results_no_standardization[ACC_CAPTIONS]:.2f}"
                        f" | Pairwise acc (images): {results_no_standardization[ACC_IMAGES]:.2f}"
                        f" | Pairwise acc (imagery): {results_no_standardization[ACC_IMAGERY]:.2f}"
                        f" | Pairwise acc (imagery whole test set): "
                        f"{results_no_standardization[ACC_IMAGERY_WHOLE_TEST]:.2f}"
                    )
                    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                    pickle.dump(results, open(results_file_path, 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--training-modes", type=str, nargs="+", default=[MODE_AGNOSTIC],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--surface", action="store_true", default=False)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--models", type=str, nargs='+', default=['imagebind'])
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--vision-features", type=str, default=SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--masks", type=str, nargs='+', default=[None])

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+', default=DEFAULT_ALPHAS)

    parser.add_argument("--n-targets-batch", type=int, default=3000)
    parser.add_argument("--n-targets-batch-refit", type=int, default=3000)
    parser.add_argument("--n-alphas-batch", type=int, default=1)

    parser.add_argument("--overwrite", action='store_true', default=False)

    parser.add_argument("--cuda", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(RIDGE_DECODER_OUT_DIR, exist_ok=True)

    run(args)
