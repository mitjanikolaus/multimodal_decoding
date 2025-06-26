import argparse
import itertools
import time
import warnings

import numpy as np
import pandas as pd
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
    calc_all_pairwise_accuracy_scores, LANG_FEAT_COMBINATION_CHOICES, standardize_latents, \
    get_fmri_data_for_splits
from data import get_latents_for_splits, SELECT_DEFAULT, LatentFeatsConfig, \
    create_null_distr_shuffled_indices, standardize_fmri_betas, SPLIT_TRAIN, MODALITY_AGNOSTIC, \
    TRAINING_MODES, ALL_SPLITS, TEST_SPLITS, NUM_STIMULI, SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS, SPLIT_IMAGERY

from utils import DEFAULT_RESOLUTION, FMRI_BETAS_DIR, DEFAULT_MODEL, ADDITIONAL_TEST_DATA_DIR, SUBJECTS_ADDITIONAL_TEST

DEFAULT_N_JOBS = 10

SEARCHLIGHT_ADDITIONAL_TEST_OUT_DIR = os.path.join(ADDITIONAL_TEST_DATA_DIR, "searchlight")
SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR = os.path.join(SEARCHLIGHT_ADDITIONAL_TEST_OUT_DIR,
                                                           "permutation_testing_results")


def train_and_test(
        estimator,
        fmri_betas_searchlight,
        latents=None,
        *,
        null_distr_dir=None,
        shuffled_indices=None,
        vertex_idx=None,
):
    estimator.fit(fmri_betas_searchlight[SPLIT_TRAIN], latents[SPLIT_TRAIN])
    predicted_latents = {split: estimator.predict(fmri_betas_searchlight[split]) for split in TEST_SPLITS}

    if null_distr_dir is not None:
        scores_null_distr = []
        for shuffle_iter in range(len(shuffled_indices[NUM_STIMULI[SPLIT_TEST_IMAGES]])):
            latents_shuffled = {split: latents[split][shuffled_indices[NUM_STIMULI[split]][shuffle_iter]] for
                                split in TEST_SPLITS}

            scores_df = calc_all_pairwise_accuracy_scores(
                latents_shuffled, predicted_latents, standardize_predictions_conds=[True]
            )
            scores_null_distr.append(scores_df)

        pickle.dump(scores_null_distr, open(os.path.join(null_distr_dir, f"{vertex_idx:010d}.p"), "wb"))

    scores = calc_all_pairwise_accuracy_scores(
        latents, predicted_latents, standardize_predictions_conds=[True]
    )

    return scores


def custom_group_iter_search_light(
        list_rows,
        vertex_indices,
        estimator,
        fmri_betas,
        latents,
        job_id,
        null_distr_dir=None,
        shuffled_indices=None,
):
    results = []
    iterator = tqdm(enumerate(list_rows), total=len(list_rows)) if job_id == 0 else enumerate(list_rows)
    for i, list_row in iterator:
        fmri_betas_searchlight = {split: betas[:, list_row] for split, betas in fmri_betas.items()}
        scores = train_and_test(
            estimator, fmri_betas_searchlight, latents,
            null_distr_dir=null_distr_dir, shuffled_indices=shuffled_indices, vertex_idx=vertex_indices[i]
        )
        scores['vertex'] = vertex_indices[i]
        results.append(scores)
    return results


def custom_search_light(
        fmri_betas,
        latents,
        estimator,
        A,
        n_jobs=-1,
        verbose=0,
        null_distr_dir=None,
        shuffled_indices=None,
):
    group_iter = GroupIterator(len(A), n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(custom_group_iter_search_light)(
                [A[i] for i in vertex_indices],
                vertex_indices,
                estimator,
                fmri_betas,
                latents,
                job_id,
                null_distr_dir,
                shuffled_indices if shuffled_indices is not None else None,
            )
            for job_id, vertex_indices in enumerate(group_iter)
        )
    scores = list(itertools.chain(*scores))
    print(f'got results for {len(scores)} vertices')
    return pd.concat(scores, ignore_index=True)


def get_adjacency_matrix(hemi, resolution=DEFAULT_RESOLUTION, nan_locations=None, radius=None, num_neighbors=None):
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
                fmri_betas, stim_ids, stim_types = get_fmri_data_for_splits(
                    subject, ALL_SPLITS, training_mode, args.betas_dir, surface=True, hemis=[hemi]
                )
                fmri_betas = standardize_fmri_betas(fmri_betas)
                for split in fmri_betas.keys():
                    print(f"{split} fMRI betas shape: {fmri_betas[split].shape}")

                feats_config = LatentFeatsConfig(
                    args.model, args.features, args.test_features, args.vision_features, args.lang_features
                )

                print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                      f"MODEL: {feats_config.model} | FEATURES: {feats_config.features}")

                latents = get_latents_for_splits(subject, feats_config, ALL_SPLITS, training_mode)
                latents = standardize_latents(latents)
                print(f"train latents shape: {latents[SPLIT_TRAIN].shape}")

                results_dir = get_results_dir(
                    feats_config, hemi, subject, training_mode, searchlight_mode_from_args(args)
                )
                os.makedirs(results_dir, exist_ok=True)

                adjacency, n_neighbors, distances = get_adjacency_matrix(
                    hemi, radius=args.radius, num_neighbors=args.n_neighbors
                )

                model = Ridge(alpha=args.l2_regularization_alpha, fit_intercept=False)

                null_distr_dir = None
                if args.create_null_distr:
                    null_distr_dir = os.path.join(results_dir, "null_distr")
                    os.makedirs(null_distr_dir, exist_ok=True)

                start = time.time()
                scores_df = custom_search_light(
                    fmri_betas, latents, estimator=model, A=adjacency, n_jobs=args.n_jobs, verbose=1,
                    null_distr_dir=null_distr_dir,
                    shuffled_indices=shuffled_indices
                )
                end = time.time()
                print(f"Searchlight time: {int(end - start)}s")

                scores_df['training_mode'] = training_mode
                scores_df['subject'] = subject
                scores_df['hemi'] = hemi

                print(scores_df)

                print(
                    f"Mean score (captions): {scores_df[scores_df.metric == SPLIT_TEST_CAPTIONS].value.mean():.2f} | "
                    f"Max score: {scores_df[scores_df.metric == SPLIT_TEST_CAPTIONS].value.max():.2f}"
                )
                print(
                    f"Mean score (images): {scores_df[scores_df.metric == SPLIT_TEST_IMAGES].value.mean():.2f} | "
                    f"Max score: {scores_df[scores_df.metric == SPLIT_TEST_IMAGES].value.max():.2f}"
                )
                print(
                    f"Mean score (imagery): {scores_df[scores_df.metric == SPLIT_IMAGERY].value.mean():.2f} | "
                    f"Max score: {scores_df[scores_df.metric == SPLIT_IMAGERY].value.max():.2f}"
                )

                results_file_path = get_results_file_path(
                    feats_config, hemi, subject, training_mode, searchlight_mode_from_args(args),
                    args.l2_regularization_alpha
                )
                scores_df.to_csv(results_file_path)


def searchlight_mode_from_args(args):
    if args.radius is not None:
        return f"radius_{args.radius}"
    elif args.n_neighbors is not None:
        return f"n_neighbors_{args.n_neighbors}"
    else:
        raise RuntimeError("Need to set either radius or n_neighbors arg!")


def get_results_dir(feats_config, hemi, subject, training_mode, mode):
    results_dir = os.path.join(
        SEARCHLIGHT_ADDITIONAL_TEST_OUT_DIR, training_mode, feats_config.model, feats_config.combined_feats,
        feats_config.vision_features, feats_config.lang_features, subject, hemi, mode
    )
    return results_dir


def get_results_file_path(feats_config, hemi, subject, training_mode, mode, l2_regularization_alpha):
    results_dir = get_results_dir(feats_config, hemi, subject, training_mode, mode)
    return os.path.join(results_dir, f"alpha_{str(l2_regularization_alpha)}.csv")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--training-modes", type=str, nargs="+", default=[MODALITY_AGNOSTIC],
                        choices=TRAINING_MODES)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS_ADDITIONAL_TEST)

    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)

    parser.add_argument("--create-null-distr", default=False, action="store_true")
    parser.add_argument("--n-permutations-per-subject", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(SEARCHLIGHT_ADDITIONAL_TEST_OUT_DIR, exist_ok=True)

    run(args)
