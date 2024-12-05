import argparse
import time
from collections import Counter

import numpy as np
import torch.cuda
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
import os
import pickle

from analyses.ridge_regression_decoding import get_fmri_data, TESTING_MODE, IMAGERY, apply_mask_and_clean, \
    standardize_fmri_betas, FEATS_SELECT_DEFAULT, get_default_features, get_default_vision_features, \
    get_default_lang_features, get_run_str, get_nn_latent_data, \
    LANG_FEAT_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, FEATURE_COMBINATION_CHOICES, TRAIN_MODE_CHOICES, \
    CAPTION, IMAGE
from utils import SUBJECTS, DEFAULT_RESOLUTION, CORR_CAPTIONS, CORR_IMAGES, CORR_ALL, RESULTS_FILE

ENCODER_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/whole_brain_encoding/")


def calc_correlation_metrics(test_fmri_betas, test_predicted_betas, stim_types):
    corr_scores_all = correlation_score(test_fmri_betas, test_predicted_betas).cpu().numpy()

    corr_scores = {CORR_ALL: corr_scores_all}
    for modality, metric_name in zip([CAPTION, IMAGE], [CORR_CAPTIONS, CORR_IMAGES]):
        preds_mod = test_predicted_betas[stim_types == modality].copy()
        targets_mod = test_fmri_betas[stim_types == modality]
        corr_scores[metric_name] = correlation_score(targets_mod, preds_mod).cpu().numpy()

    return corr_scores


def run(args):
    if args.cuda:
        print("Setting backend to cuda")
        backend = set_backend("torch_cuda")
    else:
        backend = set_backend("numpy")

    for training_mode in args.training_modes:
        for subject in args.subjects:
            train_fmri_betas_full, train_stim_ids, train_stim_types = get_fmri_data(
                subject,
                training_mode,
                surface=args.surface,
                resolution=args.resolution,
            )
            test_fmri_betas_full, test_stim_ids, test_stim_types = get_fmri_data(
                subject,
                TESTING_MODE,
                surface=args.surface,
                resolution=args.resolution,
            )
            imagery_fmri_betas_full, imagery_stim_ids, imagery_stim_types = get_fmri_data(
                subject,
                IMAGERY,
                surface=args.surface,
                resolution=args.resolution,
            )
            for mask in args.masks:
                mask = None if mask in ["none", "None"] else mask
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = apply_mask_and_clean(
                    mask, [train_fmri_betas_full, test_fmri_betas_full, imagery_fmri_betas_full], args
                )
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = standardize_fmri_betas(
                    train_fmri_betas, test_fmri_betas, imagery_fmri_betas, subject, training_mode, mask
                )

                for model_name in args.models:
                    model_name = model_name.lower()

                    for features in args.features:
                        if features == FEATS_SELECT_DEFAULT:
                            features = get_default_features(model_name)
                        test_features = args.test_features
                        if test_features == FEATS_SELECT_DEFAULT:
                            test_features = get_default_features(model_name)

                        for vision_features in args.vision_features:
                            if vision_features == FEATS_SELECT_DEFAULT:
                                vision_features = get_default_vision_features(model_name)
                            for lang_features in args.lang_features:
                                if lang_features == FEATS_SELECT_DEFAULT:
                                    lang_features = get_default_lang_features(model_name)

                                print(f"\nTRAIN MODE: {training_mode} | MASK: {mask} | SUBJECT: {subject} | "
                                      f"MODEL: {model_name} | FEATURES: {features} {vision_features} {lang_features} | "
                                      f"TEST FEATURES: {test_features}")
                                print(f"train fMRI betas shape: {train_fmri_betas.shape}")
                                print(f"test fMRI betas shape: {test_fmri_betas.shape}")
                                print(f"imagery fMRI betas shape: {imagery_fmri_betas.shape}")

                                results_dir = os.path.join(ENCODER_OUT_DIR, training_mode, subject)
                                run_str = get_run_str(
                                    model_name, features, test_features, vision_features, lang_features, mask,
                                    args.surface,
                                    args.resolution)
                                results_file_path = os.path.join(results_dir, run_str, RESULTS_FILE)
                                if os.path.isfile(results_file_path) and not args.overwrite:
                                    print(f"Skipping encoder training as results are already present at"
                                          f" {results_file_path}")
                                    continue

                                train_latents, latent_transform = get_nn_latent_data(
                                    model_name, features,
                                    vision_features,
                                    lang_features,
                                    train_stim_ids,
                                    train_stim_types,
                                    subject,
                                    training_mode,
                                )

                                model = RidgeCV(alphas=args.l2_regularization_alphas,
                                                solver_params=dict(n_targets_batch=args.n_targets_batch,
                                                                   n_alphas_batch=args.n_alphas_batch,
                                                                   n_targets_batch_refit=args.n_targets_batch_refit))

                                start = time.time()
                                model.fit(train_latents, train_fmri_betas)
                                end = time.time()
                                print(f"Elapsed time: {int(end - start)}s")

                                best_alphas = np.round(model.best_alphas_.cpu().numpy())

                                test_data_latents, _ = get_nn_latent_data(model_name, test_features,
                                                                          vision_features,
                                                                          lang_features,
                                                                          test_stim_ids,
                                                                          test_stim_types,
                                                                          subject,
                                                                          TESTING_MODE,
                                                                          nn_latent_transform=latent_transform)

                                test_predicted_betas = model.predict(test_data_latents)

                                test_fmri_betas = backend.to_numpy(test_fmri_betas)
                                test_predicted_betas = backend.to_numpy(test_predicted_betas)

                                results = {
                                    "alpha": best_alphas,
                                    "model": model_name,
                                    "subject": subject,
                                    "features": features,
                                    "test_features": test_features,
                                    "vision_features": vision_features,
                                    "lang_features": lang_features,
                                    "training_mode": training_mode,
                                    "mask": mask,
                                    "num_voxels": test_fmri_betas.shape[1],
                                    "stimulus_ids": test_stim_ids,
                                    "stimulus_types": test_stim_types,
                                    "surface": args.surface,
                                    "resolution": args.resolution,
                                }

                                results.update(
                                    calc_correlation_metrics(
                                        test_fmri_betas, test_predicted_betas, test_stim_types,
                                    )
                                )
                                print(
                                    f"Best alphas: {Counter(best_alphas)}\n"
                                    f"Corr (all): {np.mean(results[CORR_ALL]):.2f} |"
                                    f" (max: {np.max(results[CORR_ALL]):.2f})"
                                    f" | Corr (captions): {np.mean(results[CORR_CAPTIONS]):.2f} |"
                                    f" (max: {np.max(results[CORR_CAPTIONS]):.2f})"
                                    f" | Corr (images): {np.mean(results[CORR_IMAGES]):.2f} |"
                                    f" (max: {np.max(results[CORR_IMAGES]):.2f})\n\n\n"
                                )

                                os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                                pickle.dump(results, open(results_file_path, 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--surface", action="store_true", default=False)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--models", type=str, nargs='+', default=['imagebind'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--vision-features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--masks", type=str, nargs='+', default=[None])

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+',
                        default=[1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

    parser.add_argument("--n-targets-batch", type=int, default=500)
    parser.add_argument("--n-targets-batch-refit", type=int, default=100)
    parser.add_argument("--n-alphas-batch", type=int, default=1)

    parser.add_argument("--overwrite", action='store_true', default=False)

    parser.add_argument("--cuda", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(ENCODER_OUT_DIR, exist_ok=True)

    run(args)
