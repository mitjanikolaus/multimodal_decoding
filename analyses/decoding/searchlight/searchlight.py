import argparse
import gc
import sys
import time
import warnings
from collections import Counter

import numpy as np
import sklearn
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.decoding.searchlight import GroupIterator
from nilearn.surface import surface

from sklearn import neighbors
from sklearn.exceptions import ConvergenceWarning
import os
import pickle

from sklearn.linear_model import Ridge
from tqdm import tqdm

from analyses.decoding.ridge_regression_decoding import FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, \
    get_latent_features, calc_all_pairwise_accuracy_scores, LANG_FEAT_COMBINATION_CHOICES, ACC_IMAGERY, \
    ACC_IMAGERY_WHOLE_TEST, standardize_latents, tensor_pairwise_accuracy
from data import TEST_STIM_TYPES, get_fmri_surface_data, SELECT_DEFAULT, LatentFeatsConfig, create_shuffled_indices, \
    create_null_distr_shuffled_indices, standardize_fmri_betas, SPLIT_TRAIN, MODALITY_AGNOSTIC, SPLIT_TEST, \
    SPLIT_IMAGERY, \
    TRAINING_MODES
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV

from utils import SUBJECTS, DATA_DIR, DEFAULT_RESOLUTION, FMRI_BETAS_SURFACE_DIR

DEFAULT_N_JOBS = 10

SEARCHLIGHT_OUT_DIR = os.path.join(DATA_DIR, "searchlight")
SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR = os.path.join(SEARCHLIGHT_OUT_DIR, "permutation_testing_results")

DERANGEMENTS_THREE_DIMS = [[1, 2, 0], [2, 0, 1]]


def train_and_test(
        estimator,
        X,
        y=None,
        *,
        train_ids,
        test_ids,
        imagery_ids,
        null_distr_dir=None,
        shuffled_indices=None,
        list_i=None,
):
    X_train = X[train_ids]
    X_test = X[test_ids]
    X_imagery = X[imagery_ids]
    y_train = y[train_ids]
    y_test = y[test_ids]
    y_imagery = y[imagery_ids]
    estimator.fit(X_train, y_train)

    y_pred_test = estimator.predict(X_test)
    y_pred_imagery = estimator.predict(X_imagery)

    best_alphas = np.round(estimator.best_alphas_)
    print(f"Best alphas: {Counter(best_alphas)}\n")

    if null_distr_dir is not None:
        scores_null_distr = []
        for indices in shuffled_indices:
            y_test_shuffled = y_test[indices]
            shuffled_indices_imagery = DERANGEMENTS_THREE_DIMS[np.random.choice(len(DERANGEMENTS_THREE_DIMS))]
            y_imagery_shuffled = y_imagery[shuffled_indices_imagery]

            scores = calc_all_pairwise_accuracy_scores(y_test_shuffled, y_pred_test, TEST_STIM_TYPES,
                                                       y_imagery_shuffled, y_pred_imagery,
                                                       comp_cross_decoding_scores=False)
            scores_null_distr.append(scores)

        pickle.dump(scores_null_distr, open(os.path.join(null_distr_dir, f"{list_i:010d}.p"), "wb"))

    scores = calc_all_pairwise_accuracy_scores(y_test, y_pred_test, TEST_STIM_TYPES, y_imagery, y_pred_imagery)

    return scores


def custom_group_iter_search_light(
        list_rows,
        list_indices,
        estimator,
        X,
        y,
        train_ids,
        test_ids,
        imagery_ids,
        thread_id,
        null_distr_dir=None,
        shuffled_indices=None,
):
    results = []
    # t0 = time.time()
    iterator = tqdm(enumerate(list_rows), total=len(list_rows)) if thread_id == 0 else enumerate(list_rows)
    for i, list_row in iterator:
        scores = train_and_test(
            estimator, X[:, list_row], y, train_ids=train_ids, test_ids=test_ids, imagery_ids=imagery_ids,
            null_distr_dir=null_distr_dir, shuffled_indices=shuffled_indices, list_i=list_indices[i]
        )
        results.append(scores)
    return results


def custom_search_light(
        X,
        y,
        estimator,
        A,
        train_ids,
        test_ids,
        imagery_ids,
        n_jobs=-1,
        verbose=0,
        null_distr_dir=None,
        shuffled_indices=None,
):
    group_iter = GroupIterator(len(A), n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(custom_group_iter_search_light)(
                [A[i] for i in list_i],
                list_i,
                estimator,
                X,
                y,
                train_ids,
                test_ids,
                imagery_ids,
                thread_id,
                null_distr_dir,
                shuffled_indices if shuffled_indices is not None else None,
            )
            for thread_id, list_i in enumerate(group_iter)
        )
    return np.concatenate(scores)


def get_adjacency_matrix(hemi, resolution, nan_locations=None, radius=None, num_neighbors=None):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)

    infl_mesh = fsaverage[f"infl_{hemi}"]
    coords, _ = surface.load_surf_mesh(infl_mesh)
    if nan_locations is not None:
        coords = coords[~nan_locations]

    nn = neighbors.NearestNeighbors(radius=radius)

    nearest_neighbors = None
    distances = None
    if radius is not None:
        adjacency = [np.argwhere(arr == 1)[:, 0] for arr in
                     nn.fit(coords).radius_neighbors_graph(coords).toarray()]
        nearest_neighbors = [len(adj) for adj in adjacency]

        print(
            f"Number of neighbors within {radius}mm radius: {np.mean(nearest_neighbors):.1f} "
            f"(max: {np.max(nearest_neighbors):.0f} | min: {np.min(nearest_neighbors):.0f})")
    elif num_neighbors is not None:
        distances, adjacency = nn.fit(coords).kneighbors(coords, n_neighbors=num_neighbors)
        print(f"Max radius {num_neighbors} neighbors: {distances.max():.2f}mm")
        print(f"Mean radius: {distances.max(axis=1).mean():.2f}mm")
    else:
        raise RuntimeError("Need to set either radius or n_neighbors arg!")
    return adjacency, nearest_neighbors, distances


def run(args):
    shuffled_indices = None
    if args.create_null_distr:
        shuffled_indices = create_null_distr_shuffled_indices(args.n_permutations_per_subject)

    for subject in args.subjects:
        for training_mode in args.training_modes:
            for hemi in args.hemis:
                train_fmri, train_stim_ids, train_stim_types = get_fmri_surface_data(
                    args.betas_dir, subject, SPLIT_TRAIN, training_mode, args.resolution, hemi
                )
                test_fmri, test_stim_ids, test_stim_types = get_fmri_surface_data(
                    args.betas_dir, subject, SPLIT_TEST, resolution=args.resolution, hemi=hemi
                )
                imagery_fmri, imagery_stim_ids, imagery_stim_types = get_fmri_surface_data(
                    args.betas_dir, subject, SPLIT_IMAGERY, resolution=args.resolution, hemi=hemi
                )
                nan_locations = np.isnan(train_fmri[0])
                train_fmri, test_fmri, imagery_fmri = standardize_fmri_betas(train_fmri, test_fmri, imagery_fmri)

                feats_config = LatentFeatsConfig(
                    args.model, args.features, args.test_features, args.vision_features, args.lang_features
                )

                print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                      f"MODEL: {feats_config.model} | FEATURES: {feats_config.features}")

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

                latents = np.concatenate((train_latents, test_latents, imagery_latents))

                results_dir = get_results_dir(
                    feats_config, hemi, subject, training_mode, args.resolution, searchlight_mode_from_args(args)
                )
                os.makedirs(results_dir, exist_ok=True)

                print("Hemisphere: ", hemi)
                print(f"train_fmri shape: {train_fmri.shape}")
                print(f"test_fmri shape: {test_fmri.shape}")
                print(f"imagery_fmri shape: {imagery_fmri.shape}")

                train_ids = list(range(len(train_fmri)))
                test_ids = list(range(len(train_fmri), len(train_fmri) + len(test_fmri)))
                imagery_ids = list(range(len(train_ids) + len(test_ids),
                                         len(train_ids) + len(test_ids) + len(imagery_fmri)))

                X = np.concatenate((train_fmri, test_fmri, imagery_fmri))

                adjacency, n_neighbors, distances = get_adjacency_matrix(
                    hemi, args.resolution, nan_locations, args.radius, args.n_neighbors
                )

                # model = Ridge(alpha=args.l2_regularization_alpha)
                model = RidgeCV(
                    cv=5,
                    alphas=args.l2_regularization_alphas,
                    solver_params=dict(
                        n_targets_batch=128,
                        n_alphas_batch=1,
                        n_targets_batch_refit=128,
                        score_func=tensor_pairwise_accuracy,
                    )
                )
                # skip input data checking to limit memory use
                # (https://gallantlab.org/himalaya/troubleshooting.html?highlight=cuda)
                sklearn.set_config(assume_finite=True)

                null_distr_dir = None
                if args.create_null_distr:
                    null_distr_dir = os.path.join(results_dir, "null_distr")
                    os.makedirs(null_distr_dir, exist_ok=True)

                X = X.astype(np.float16)
                latents = latents.astype(np.float16)

                start = time.time()
                scores = custom_search_light(
                    X, latents, estimator=model, A=adjacency, train_ids=train_ids, test_ids=test_ids,
                    imagery_ids=imagery_ids, n_jobs=args.n_jobs, verbose=1, null_distr_dir=null_distr_dir,
                    shuffled_indices=shuffled_indices
                )
                end = time.time()
                print(f"Searchlight time: {int(end - start)}s")
                test_scores_caps = [score["pairwise_acc_captions"] for score in scores]
                print(
                    f"Mean score (captions): {np.mean(test_scores_caps):.2f} | "
                    f"Max score: {np.max(test_scores_caps):.2f}"
                )

                test_scores_imgs = [score["pairwise_acc_images"] for score in scores]
                print(
                    f"Mean score (images): {np.mean(test_scores_imgs):.2f} | "
                    f"Max score: {np.max(test_scores_imgs):.2f}"
                )

                imagery_scores = [score[ACC_IMAGERY] for score in scores]
                print(
                    f"Mean score ({ACC_IMAGERY}): {np.mean(imagery_scores):.2f} | "
                    f"Max score: {np.max(imagery_scores):.2f}"
                )

                imagery_whole_test_set_scores = [score[ACC_IMAGERY_WHOLE_TEST] for score in scores]
                print(
                    f"Mean score ({ACC_IMAGERY_WHOLE_TEST}): {np.mean(imagery_whole_test_set_scores):.2f} | "
                    f"Max score: {np.max(imagery_whole_test_set_scores):.2f}"
                )

                results_dict = {
                    "nan_locations": nan_locations,
                    "adjacency": adjacency,
                    "n_neighbors": n_neighbors,
                    "distances": distances,
                    "scores": scores,
                }
                results_file_path = get_results_file_path(
                    feats_config, hemi, subject, training_mode, args.resolution, searchlight_mode_from_args(args),
                    args.l2_regularization_alpha
                )
                pickle.dump(results_dict, open(results_file_path, 'wb'))

                del X, latents, train_fmri
                gc.collect()


def searchlight_mode_from_args(args):
    if args.radius is not None:
        return f"radius_{args.radius}"
    elif args.n_neighbors is not None:
        return f"n_neighbors_{args.n_neighbors}"
    else:
        raise RuntimeError("Need to set either radius or n_neighbors arg!")


def get_results_dir(feats_config, hemi, subject, training_mode, resolution, mode):
    results_dir = os.path.join(
        SEARCHLIGHT_OUT_DIR, training_mode, feats_config.model, feats_config.combined_feats, subject, resolution,
        hemi, mode
    )
    return results_dir


def get_results_file_path(feats_config, hemi, subject, training_mode, resolution, mode, l2_regularization_alpha):
    results_dir = get_results_dir(feats_config, hemi, subject, training_mode, resolution, mode)
    return os.path.join(results_dir, f"alpha_{str(l2_regularization_alpha)}.p")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_SURFACE_DIR)

    parser.add_argument("--training-modes", type=str, nargs="+", default=[MODALITY_AGNOSTIC],
                        choices=TRAINING_MODES)

    parser.add_argument("--model", type=str, default="imagebind")
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    parser.add_argument("--l2-regularization-alphas", type=float, nargs="+", default=[1e-1,1,1e2,1e4])

    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)

    parser.add_argument("--create-null-distr", default=False, action="store_true")
    parser.add_argument("--n-permutations-per-subject", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(SEARCHLIGHT_OUT_DIR, exist_ok=True)

    run(args)
