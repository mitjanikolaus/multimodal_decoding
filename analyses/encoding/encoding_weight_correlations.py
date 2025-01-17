import argparse
import time
from collections import Counter

import numpy as np
import sklearn
from scipy.stats import pearsonr
from tqdm import tqdm

from analyses.encoding.ridge_regression_encoding import get_results_file_path, get_null_distr_results_path, \
    calc_correlation_metrics
from data import get_fmri_surface_data, SELECT_DEFAULT, LatentFeatsConfig, create_shuffled_indices, \
    create_null_distr_seeds
from eval import CORR_ALL, CORR_CAPTIONS, CORR_IMAGES
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
import os
import pickle

from analyses.decoding.ridge_regression_decoding import TESTING_MODE, standardize_fmri_betas, \
    get_latent_features, \
    LANG_FEAT_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, FEATURE_COMBINATION_CHOICES, \
    standardize_latents
from utils import SUBJECTS, DEFAULT_RESOLUTION, HEMIS, FMRI_BETAS_DIR, MOD_SPECIFIC_IMAGES, MOD_SPECIFIC_CAPTIONS


def calc_feats_corr(subject, args):
    feats_config = LatentFeatsConfig(
        args.model, args.features, args.test_features, args.vision_features, args.lang_features
    )

    weights = dict()
    for training_mode in [MOD_SPECIFIC_CAPTIONS, MOD_SPECIFIC_IMAGES]:
        weights[training_mode] = []
        for hemi in ['left']:#HEMIS:
            results_file_path = get_results_file_path(
                subject, training_mode, feats_config, args.resolution, hemi
            )
            results = pickle.load(open(results_file_path, "rb"))
            weights[training_mode].append(results['weights'])
        weights[training_mode] = np.hstack(weights[training_mode])

    corrs = []
    pvals = []
    for weights_mod_spec_imgs, weights_mod_spec_caps in zip(weights[MOD_SPECIFIC_IMAGES], weights[MOD_SPECIFIC_CAPTIONS]):
        corr = pearsonr(weights_mod_spec_imgs, weights_mod_spec_caps)
        corrs.append(corr[0])
        pvals.append(corr[1])

    return np.array(corrs)


def run(args):
    if args.cuda:
        print("Setting backend to cuda")
        backend = set_backend("torch_cuda")
    else:
        backend = set_backend("numpy")

    random_seeds = create_null_distr_seeds(args.n_permutations_per_subject) if args.create_null_distr else None

    for subject in args.subjects:
        corrs = calc_feats_corr(subject, args)
        for hemi in HEMIS:
            for training_mode in [MOD_SPECIFIC_CAPTIONS, MOD_SPECIFIC_IMAGES]:
                train_fmri_betas_full, train_stim_ids, train_stim_types = get_fmri_surface_data(
                    subject,
                    training_mode,
                    resolution=args.resolution,
                    hemis=[hemi],
                )
                test_fmri_betas_full, test_stim_ids, test_stim_types = get_fmri_surface_data(
                    subject,
                    TESTING_MODE,
                    resolution=args.resolution,
                    hemis=[hemi]
                )
                train_fmri_betas = train_fmri_betas_full[hemi]
                test_betas = test_fmri_betas_full[hemi]

                nan_locations = np.isnan(train_fmri_betas[0])
                train_fmri_betas = train_fmri_betas[:, ~nan_locations]
                test_betas = test_betas[:, ~nan_locations]

                train_fmri_betas, test_betas = standardize_fmri_betas(
                    train_fmri_betas, test_betas
                )

                feats_config = LatentFeatsConfig(
                    args.model, args.features, args.test_features, args.vision_features, args.lang_features
                )

                train_latents = get_latent_features(args.model, feats_config, train_stim_ids, train_stim_types)
                test_latents = get_latent_features(
                    args.model, feats_config, test_stim_ids, test_stim_types, test_mode=True
                )
                print(f"Number of feats after corr threshold filtering: {np.sum(corrs > args.corr_threshold)}")
                train_latents = train_latents[:, corrs > args.corr_threshold]
                test_latents = test_latents[:, corrs > args.corr_threshold]

                train_latents, test_latents = standardize_latents(train_latents, test_latents)

                print(f"\nTRAIN MODE: {training_mode} | HEMI: {hemi} | SUBJECT: {subject} | "
                      f"MODEL: {args.model} | FEATURES: {feats_config.features} "
                      f"{feats_config.vision_features} {feats_config.lang_features} | "
                      f"TEST FEATURES: {feats_config.test_features}")
                print(f"train fMRI betas shape: {train_fmri_betas.shape}")
                print(f"test fMRI betas shape: {test_betas.shape}")

                results_file_path = get_results_file_path(
                    subject, training_mode, feats_config, args.resolution, hemi, corr_threshold=args.corr_threshold
                )
                if os.path.isfile(results_file_path) and not args.overwrite:
                    print(f"Skipping encoder training as results are already present at"
                          f" {results_file_path}")
                    continue

                clf = RidgeCV(
                    alphas=args.l2_regularization_alphas,
                    solver_params=dict(
                        n_targets_batch=args.n_targets_batch,
                        n_alphas_batch=args.n_alphas_batch,
                        n_targets_batch_refit=args.n_targets_batch_refit
                    )
                )

                start = time.time()
                train_latents = train_latents.astype(np.float32)
                train_fmri_betas = train_fmri_betas.astype(np.float32)

                # skip input data checking to limit memory use
                # (https://gallantlab.org/himalaya/troubleshooting.html?highlight=cuda)
                sklearn.set_config(assume_finite=True)

                clf.fit(train_latents, train_fmri_betas)
                end = time.time()
                print(f"Elapsed time: {int(end - start)}s")

                best_alphas = np.round(backend.to_numpy(clf.best_alphas_))

                test_latents = test_latents.astype(np.float32)
                test_predicted_betas = clf.predict(test_latents)

                test_betas = backend.to_numpy(test_betas)
                test_predicted_betas = backend.to_numpy(test_predicted_betas)

                results = {
                    "alpha": best_alphas,
                    "model": args.model,
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
                    f" (max: {np.max(results[CORR_IMAGES]):.2f})\n\n\n"
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

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--model", type=str, default='imagebind')

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

    parser.add_argument("--n-targets-batch", type=int, default=500)
    parser.add_argument("--n-targets-batch-refit", type=int, default=100)
    parser.add_argument("--n-alphas-batch", type=int, default=2)

    parser.add_argument("--overwrite", action='store_true', default=False)

    parser.add_argument("--cuda", action='store_true', default=False)

    parser.add_argument("--create-null-distr", default=False, action="store_true")
    parser.add_argument("--n-permutations-per-subject", type=int, default=100)

    parser.add_argument("--corr-threshold", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
