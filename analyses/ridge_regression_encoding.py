import argparse
import time
from collections import Counter

import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
import os
import pickle

from analyses.ridge_regression_decoding import get_fmri_data, TESTING_MODE, standardize_fmri_betas, \
    FEATS_SELECT_DEFAULT, get_default_features, get_default_vision_features, \
    get_default_lang_features, get_run_str, get_nn_latent_data, \
    LANG_FEAT_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, FEATURE_COMBINATION_CHOICES, TRAIN_MODE_CHOICES, \
    CAPTION, IMAGE
from utils import SUBJECTS, DEFAULT_RESOLUTION, CORR_CAPTIONS, CORR_IMAGES, CORR_ALL, RESULTS_FILE, HEMIS, \
    CORR_CROSS_IMAGES_TO_CAPTIONS, CORR_CROSS_CAPTIONS_TO_IMAGES, FMRI_BETAS_DIR, DATA_DIR, create_null_distr_seeds, \
    create_shuffled_indices

ENCODER_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/whole_brain_encoding/")
ENCODING_RESULTS_DIR = os.path.join(DATA_DIR, "encoding_results")


def calc_correlation_metrics(test_fmri_betas, test_predicted_betas, stim_types):
    corr_scores_all = correlation_score(test_fmri_betas, test_predicted_betas).cpu().numpy()

    corr_scores = {CORR_ALL: corr_scores_all}
    for modality, metric_name in zip([CAPTION, IMAGE], [CORR_CAPTIONS, CORR_IMAGES]):
        preds_mod = test_predicted_betas[stim_types == modality].copy()
        targets_mod = test_fmri_betas[stim_types == modality]
        corr_scores[metric_name] = correlation_score(targets_mod, preds_mod).cpu().numpy()

    for mod_preds, mod_targets, metric_name in zip([CAPTION, IMAGE], [IMAGE, CAPTION],
                                                   [CORR_CROSS_CAPTIONS_TO_IMAGES, CORR_CROSS_IMAGES_TO_CAPTIONS]):
        preds_mod = test_predicted_betas[stim_types == mod_preds].copy()
        targets_mod = test_fmri_betas[stim_types == mod_targets]
        corr_scores[metric_name] = correlation_score(targets_mod, preds_mod).cpu().numpy()

    return corr_scores


def run(args):
    if args.cuda:
        print("Setting backend to cuda")
        backend = set_backend("torch_cuda")
    else:
        backend = set_backend("numpy")

    random_seeds = create_null_distr_seeds(args.num_permuations_per_subject) if args.create_null_distr else None

    for training_mode in args.training_modes:
        for subject in args.subjects:
            train_fmri_betas_full, train_stim_ids, train_stim_types = get_fmri_data(
                args.betas_dir,
                subject,
                training_mode,
                surface=True,
                resolution=args.resolution,
            )
            test_fmri_betas_full, test_stim_ids, test_stim_types = get_fmri_data(
                args.betas_dir,
                subject,
                TESTING_MODE,
                surface=True,
                resolution=args.resolution,
            )
            for hemi in HEMIS:
                train_fmri_betas = train_fmri_betas_full[hemi]
                test_fmri_betas = test_fmri_betas_full[hemi]

                train_fmri_betas = np.nan_to_num(train_fmri_betas)
                test_fmri_betas = np.nan_to_num(test_fmri_betas)

                train_fmri_betas, test_fmri_betas, _ = standardize_fmri_betas(
                    train_fmri_betas, test_fmri_betas, imagery_fmri_betas=None, subject=subject,
                    training_mode=training_mode, mask_name=None
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

                                print(f"\nTRAIN MODE: {training_mode} | HEMI: {hemi} | SUBJECT: {subject} | "
                                      f"MODEL: {model_name} | FEATURES: {features} {vision_features} {lang_features} | "
                                      f"TEST FEATURES: {test_features}")
                                print(f"train fMRI betas shape: {train_fmri_betas.shape}")
                                print(f"test fMRI betas shape: {test_fmri_betas.shape}")

                                results_dir = os.path.join(ENCODER_OUT_DIR, training_mode, subject)
                                run_str = get_run_str(
                                    model_name, features, test_features, vision_features, lang_features, mask=None,
                                    surface=True, resolution=args.resolution, hemi=hemi)
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

                                best_alphas = np.round(backend.to_numpy(model.best_alphas_))

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
                                    "hemi": hemi,
                                    "num_voxels": test_fmri_betas.shape[1],
                                    "stimulus_ids": test_stim_ids,
                                    "stimulus_types": test_stim_types,
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

                                if args.create_null_distr:
                                    scores_null_distr = []
                                    for seed in random_seeds:
                                        shuffled_indices = create_shuffled_indices(seed)
                                        test_fmri_betas_shuffled = test_fmri_betas[shuffled_indices]

                                        scores = calc_correlation_metrics(
                                            test_fmri_betas_shuffled, test_predicted_betas, test_stim_types,
                                        )
                                        scores_null_distr.append(scores)

                                    pickle.dump(
                                        scores_null_distr,
                                        open(os.path.join(os.path.dirname(results_file_path), f"null_distr.p"), "wb")
                                    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

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

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+',
                        default=[1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

    parser.add_argument("--n-targets-batch", type=int, default=500)
    parser.add_argument("--n-targets-batch-refit", type=int, default=100)
    parser.add_argument("--n-alphas-batch", type=int, default=2)

    parser.add_argument("--overwrite", action='store_true', default=False)

    parser.add_argument("--cuda", action='store_true', default=False)

    parser.add_argument("--create-null-distr", default=False, action="store_true")
    parser.add_argument("--n-permutations-per-subject", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(ENCODER_OUT_DIR, exist_ok=True)

    run(args)
