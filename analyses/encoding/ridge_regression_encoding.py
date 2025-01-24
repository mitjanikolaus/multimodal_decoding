import argparse
import time
from collections import Counter

import numpy as np
import sklearn
from tqdm import tqdm

from data import CAPTION, IMAGE, get_fmri_surface_data, SELECT_DEFAULT, LatentFeatsConfig, create_shuffled_indices, \
    create_null_distr_seeds
from eval import CORR_ALL, CORR_CAPTIONS, CORR_IMAGES, CORR_CROSS_CAPTIONS_TO_IMAGES, CORR_CROSS_IMAGES_TO_CAPTIONS
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV, GroupRidgeCV
from himalaya.scoring import correlation_score
import os
import pickle

from analyses.decoding.ridge_regression_decoding import TESTING_MODE, standardize_fmri_betas, \
    get_latent_features, \
    LANG_FEAT_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, FEATURE_COMBINATION_CHOICES, TRAIN_MODE_CHOICES, \
    standardize_latents
from utils import SUBJECTS, DEFAULT_RESOLUTION, RESULTS_FILE, HEMIS, FMRI_BETAS_DIR, DATA_DIR

ENCODER_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/whole_brain_encoding/")
ENCODING_RESULTS_DIR = os.path.join(DATA_DIR, "encoding_results")


def calc_correlation_metrics(test_fmri_betas, test_predicted_betas, stim_types, backend):
    corr_scores_all = backend.to_numpy(correlation_score(test_fmri_betas, test_predicted_betas))

    corr_scores = {CORR_ALL: corr_scores_all}
    for modality, metric_name in zip([CAPTION, IMAGE], [CORR_CAPTIONS, CORR_IMAGES]):
        preds_mod = test_predicted_betas[stim_types == modality].copy()
        targets_mod = test_fmri_betas[stim_types == modality]
        corr_scores[metric_name] = backend.to_numpy(correlation_score(targets_mod, preds_mod))

    for mod_preds, mod_targets, metric_name in zip([CAPTION, IMAGE], [IMAGE, CAPTION],
                                                   [CORR_CROSS_CAPTIONS_TO_IMAGES, CORR_CROSS_IMAGES_TO_CAPTIONS]):
        preds_mod = test_predicted_betas[stim_types == mod_preds].copy()
        targets_mod = test_fmri_betas[stim_types == mod_targets]
        corr_scores[metric_name] = backend.to_numpy(correlation_score(targets_mod, preds_mod))

    return corr_scores


def get_run_str(feats_config, resolution, hemi):
    run_str = f"{feats_config.model}_{feats_config.combined_feats}"
    run_str += f"_{feats_config.vision_features}"
    run_str += f"_{feats_config.lang_features}"
    run_str += f"_surface_{resolution}"
    run_str += f"_{hemi}_hemi"
    return run_str


def get_results_file_path(
        subject, training_mode, feats_config, resolution, hemi, corr_threshold=None, add_gabor_feats=False
):
    run_str = get_run_str(feats_config, resolution, hemi=hemi)
    if corr_threshold is not None:
        run_str += f"_corr_thresh_{corr_threshold}"
    if add_gabor_feats:
        run_str += f"_gabor"
    results_file_path = os.path.join(ENCODER_OUT_DIR, training_mode, subject, run_str, RESULTS_FILE)
    return results_file_path


def get_null_distr_results_path(subject, training_mode, feats_config, resolution, hemi):
    run_str = get_run_str(feats_config, resolution, hemi=hemi)
    results_file_path = os.path.join(ENCODER_OUT_DIR, training_mode, subject, run_str, "null_distr.p")
    return results_file_path


def run(args):
    if args.cuda:
        print("Setting backend to cuda")
        backend = set_backend("torch_cuda")
    else:
        backend = set_backend("numpy")

    random_seeds = create_null_distr_seeds(args.n_permutations_per_subject) if args.create_null_distr else None

    for training_mode in args.training_modes:
        for subject in args.subjects:
            for hemi in HEMIS:
                train_fmri_betas, train_stim_ids, train_stim_types = get_fmri_surface_data(
                    subject,
                    training_mode,
                    resolution=args.resolution,
                    hemi=hemi,
                )
                test_betas, test_stim_ids, test_stim_types = get_fmri_surface_data(
                    subject,
                    TESTING_MODE,
                    resolution=args.resolution,
                    hemi=hemi,
                )
                nan_locations = np.isnan(train_fmri_betas[0])
                train_fmri_betas, test_betas = standardize_fmri_betas(train_fmri_betas, test_betas)

                for model in args.models:
                    feats_config = LatentFeatsConfig(
                        model, args.features, args.test_features, args.vision_features, args.lang_features
                    )

                    train_latents = get_latent_features(feats_config, train_stim_ids, train_stim_types)
                    test_latents = get_latent_features(
                        feats_config, test_stim_ids, test_stim_types, test_mode=True
                    )
                    train_latents, test_latents = standardize_latents(train_latents, test_latents)

                    print(f"\nTRAIN MODE: {training_mode} | HEMI: {hemi} | SUBJECT: {subject} | "
                          f"MODEL: {model} | FEATURES: {feats_config.features} "
                          f"{feats_config.vision_features} {feats_config.lang_features} | "
                          f"TEST FEATURES: {feats_config.test_features}")
                    print(f"train fMRI betas shape: {train_fmri_betas.shape}")
                    print(f"test fMRI betas shape: {test_betas.shape}")

                    results_file_path = get_results_file_path(
                        subject, training_mode, feats_config, args.resolution, hemi
                    )
                    if os.path.isfile(results_file_path) and not args.overwrite:
                        print(f"Skipping encoder training as results are already present at"
                              f" {results_file_path}")
                        continue

                    train_latents = train_latents.astype(np.float32)
                    test_latents = test_latents.astype(np.float32)
                    train_fmri_betas = train_fmri_betas.astype(np.float32)

                    # skip input data checking to limit memory use
                    # (https://gallantlab.org/himalaya/troubleshooting.html?highlight=cuda)
                    sklearn.set_config(assume_finite=True)

                    if args.add_gabor_feats:
                        feats_config = LatentFeatsConfig("gabor")
                        train_latents_gabor = get_latent_features(feats_config, train_stim_ids, train_stim_types)
                        test_latents_gabor = get_latent_features(
                            feats_config, test_stim_ids, test_stim_types, test_mode=True
                        )
                        train_latents_gabor, test_latents_gabor = standardize_latents(
                            train_latents_gabor, test_latents_gabor
                        )
                        train_latents_gabor = train_latents_gabor.astype(np.float32)
                        test_latents_gabor = test_latents_gabor.astype(np.float32)

                        groups = np.concatenate(
                            (np.repeat(0, len(train_latents[0])), np.repeat(1, len(train_latents_gabor[0])))
                        )
                        clf = GroupRidgeCV(
                            groups=groups,
                            solver_params=dict(
                                n_iter=10,
                                alphas=np.array(args.l2_regularization_alphas),
                                n_targets_batch=args.n_targets_batch,
                                n_alphas_batch=args.n_alphas_batch,
                                n_targets_batch_refit=args.n_targets_batch_refit
                            )
                        )

                        clf.fit(np.hstack((train_latents, train_latents_gabor)), train_fmri_betas)

                        best_alphas = np.round(backend.to_numpy(clf.best_alphas_))

                        test_latents_gabor = np.zeros_like(test_latents_gabor)
                        test_predicted_betas = clf.predict(np.hstack((test_latents, test_latents_gabor)))

                    else:
                        clf = RidgeCV(
                            alphas=args.l2_regularization_alphas,
                            solver_params=dict(
                                n_targets_batch=args.n_targets_batch,
                                n_alphas_batch=args.n_alphas_batch,
                                n_targets_batch_refit=args.n_targets_batch_refit
                            )
                        )

                        clf.fit(train_latents, train_fmri_betas)

                        best_alphas = np.round(backend.to_numpy(clf.best_alphas_))

                        test_predicted_betas = clf.predict(test_latents)

                    test_betas = backend.to_numpy(test_betas)
                    test_predicted_betas = backend.to_numpy(test_predicted_betas)

                    results = {
                        "alpha": best_alphas,
                        "model": model,
                        "subject": subject,
                        "features": feats_config.features,
                        "test_features": feats_config.test_features,
                        "vision_features": feats_config.vision_features,
                        "lang_features": feats_config.lang_features,
                        "training_mode": training_mode,
                        "hemi": hemi,
                        "num_voxels": test_betas.shape[1],
                        "stimulus_ids": test_stim_ids,
                        "stimulus_types": test_stim_types,
                        "resolution": args.resolution,
                        "nan_locations": nan_locations,
                        "test_predicted_betas": test_predicted_betas,
                        "test_betas": test_betas,
                        "weights": clf.coef_,
                    }
                    scores = calc_correlation_metrics(
                        test_betas, test_predicted_betas, test_stim_types, backend
                    )
                    results.update(scores)
                    print(
                        f"Best alphas: {Counter(best_alphas)}\n"
                        f"Corr (all): {np.mean(results[CORR_ALL]):.2f} |"
                        f" (max: {np.max(results[CORR_ALL]):.2f})"
                        f" | Corr (captions): {np.mean(results[CORR_CAPTIONS]):.2f} |"
                        f" (max: {np.max(results[CORR_CAPTIONS]):.2f})"
                        f" | Corr (images): {np.mean(results[CORR_IMAGES]):.2f} |"
                        f" (max: {np.max(results[CORR_IMAGES]):.2f})\n"
                        f"Corr (captions, pos only): {np.mean(results[CORR_CAPTIONS][results[CORR_CAPTIONS] > 0]):.2f} |"
                        f" Corr (images, pos only): {np.mean(results[CORR_IMAGES][results[CORR_IMAGES] > 0]):.2f}\n"
                        f"Num vertices positive corr (captions): {np.sum(results[CORR_CAPTIONS] > 0)}/{len(results[CORR_CAPTIONS])} |"
                        f" Num vertices positive corr (images): {np.sum(results[CORR_IMAGES] > 0)}/{len(results[CORR_IMAGES])}\n\n"
                    )

                    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                    pickle.dump(results, open(results_file_path, 'wb'))

                    if args.create_null_distr:
                        scores_null_distr = []
                        for seed in tqdm(random_seeds, desc="creating null distribution"):
                            shuffled_indices = create_shuffled_indices(seed)
                            test_fmri_betas_shuffled = test_betas[shuffled_indices]

                            scores = calc_correlation_metrics(
                                test_fmri_betas_shuffled, test_predicted_betas, test_stim_types, backend
                            )
                            scores_null_distr.append(scores)

                        null_distr_file_path = get_null_distr_results_path(
                            subject, training_mode, feats_config, args.resolution, hemi
                        )
                        pickle.dump(scores_null_distr, open(null_distr_file_path, "wb"))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

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

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+',
                        default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

    parser.add_argument("--n-targets-batch", type=int, default=5000)
    parser.add_argument("--n-targets-batch-refit", type=int, default=5000)
    parser.add_argument("--n-alphas-batch", type=int, default=2)

    parser.add_argument("--overwrite", action='store_true', default=False)

    parser.add_argument("--cuda", action='store_true', default=False)

    parser.add_argument("--create-null-distr", default=False, action="store_true")
    parser.add_argument("--n-permutations-per-subject", type=int, default=100)

    parser.add_argument("--add-gabor-feats", default=False, action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(ENCODER_OUT_DIR, exist_ok=True)

    run(args)
