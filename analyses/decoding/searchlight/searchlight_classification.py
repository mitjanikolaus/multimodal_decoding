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

from tqdm import tqdm

from analyses.decoding.ridge_regression_decoding import get_fmri_data_for_splits
from data import create_null_distr_shuffled_indices, MODALITY_AGNOSTIC, \
    ATTENTION_MOD_SPLITS, TEST_IMAGES_UNATTENDED, TEST_CAPTIONS_UNATTENDED, TEST_IMAGES_ATTENDED, TEST_CAPTIONS_ATTENDED
from eval import pairwise_accuracy

from utils import DEFAULT_RESOLUTION, FMRI_BETAS_DIR, ADDITIONAL_TEST_DATA_DIR, SUBJECTS_ADDITIONAL_TEST, \
    HEMIS

DEFAULT_N_JOBS = 10

SEARCHLIGHT_ADDITIONAL_TEST_CLASSIFICATION_OUT_DIR = os.path.join(ADDITIONAL_TEST_DATA_DIR,
                                                                  "searchlight_classification")
SEARCHLIGHT_CLASSIFICATION_PERMUTATION_TESTING_RESULTS_DIR = os.path.join(
    SEARCHLIGHT_ADDITIONAL_TEST_CLASSIFICATION_OUT_DIR,
    "permutation_testing_results")


def train_and_test(
        fmri_betas_searchlight,
        *,
        null_distr_dir=None,
        shuffled_indices=None,
        vertex_idx=None,
):
    scores = []
    for training_split in ATTENTION_MOD_SPLITS:
        fmri_betas_train = fmri_betas_searchlight[training_split]

        testing_splits = [split for split in ATTENTION_MOD_SPLITS if split != training_split]
        for testing_split in testing_splits:
            fmri_betas_test = fmri_betas_searchlight[testing_split]
            # print(f"testing set shape: {fmri_betas_test.shape} (split: {testing_split})")
            pairwise_acc = pairwise_accuracy(fmri_betas_train, fmri_betas_test, standardize_predictions=False,
                                             standardize_latents=False)
            result = {
                'value': pairwise_acc,
                'train_split': training_split,
                'test_split': testing_split,
            }

            # print(result)
            scores.append(result)

    # if null_distr_dir is not None:
    #     scores_null_distr = []
    #     for shuffle_iter in range(len(shuffled_indices[NUM_STIMULI[TEST_IMAGES]])):
    #         latents_shuffled = {split: latents[split][shuffled_indices[NUM_STIMULI[split]][shuffle_iter]] for
    #                             split in TEST_SPLITS}
    #
    #         scores_df = calc_all_pairwise_accuracy_scores(
    #             latents_shuffled, predicted_latents, standardize_predictions_conds=[True]
    #         )
    #         scores_null_distr.append(scores_df)
    #
    #     pickle.dump(scores_null_distr, open(os.path.join(null_distr_dir, f"{vertex_idx:010d}.p"), "wb"))
    scores = pd.DataFrame(scores)
    return scores


def custom_group_iter_search_light(
        list_rows,
        vertex_indices,
        fmri_betas,
        job_id,
        null_distr_dir=None,
        shuffled_indices=None,
):
    results = []
    iterator = tqdm(enumerate(list_rows), total=len(list_rows)) if job_id == 0 else enumerate(list_rows)
    for i, list_row in iterator:
        fmri_betas_searchlight = {split: betas[:, list_row] for split, betas in fmri_betas.items()}
        scores = train_and_test(
            fmri_betas_searchlight,
            null_distr_dir=null_distr_dir, shuffled_indices=shuffled_indices, vertex_idx=vertex_indices[i]
        )
        scores['vertex'] = vertex_indices[i]
        results.append(scores)
    return results


def custom_search_light(
        fmri_betas,
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
                fmri_betas,
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
        for hemi in HEMIS:
            fmri_betas, stim_ids, stim_types = get_fmri_data_for_splits(
                subject, ATTENTION_MOD_SPLITS, MODALITY_AGNOSTIC, args.betas_dir, surface=True, hemis=[hemi]
            )
            # fmri_betas = standardize_fmri_betas(fmri_betas)
            for split in fmri_betas.keys():
                print(f"{split} fMRI betas shape: {fmri_betas[split].shape}")

            print(f"\nSUBJECT: {subject} | HEMI: {hemi}")

            results_dir = get_results_dir(
                hemi, subject, searchlight_mode_from_args(args)
            )
            os.makedirs(results_dir, exist_ok=True)

            adjacency, n_neighbors, distances = get_adjacency_matrix(
                hemi, radius=args.radius, num_neighbors=args.n_neighbors
            )

            null_distr_dir = None
            if args.create_null_distr:
                null_distr_dir = os.path.join(results_dir, "null_distr")
                os.makedirs(null_distr_dir, exist_ok=True)

            start = time.time()
            scores_df = custom_search_light(
                fmri_betas, A=adjacency, n_jobs=args.n_jobs, verbose=1,
                null_distr_dir=null_distr_dir,
                shuffled_indices=shuffled_indices
            )
            end = time.time()
            print(f"Searchlight time: {int(end - start)}s")

            scores_df['subject'] = subject
            scores_df['hemi'] = hemi

            print(scores_df)

            print(
                f"Mean score (captions attended -> captions unattended): {scores_df[(scores_df.train_split == TEST_CAPTIONS_ATTENDED) & (scores_df.test_split == TEST_CAPTIONS_UNATTENDED)].value.mean():.2f} | "
                f"Max score: {scores_df[(scores_df.train_split == TEST_CAPTIONS_ATTENDED) & (scores_df.test_split == TEST_CAPTIONS_UNATTENDED)].value.max():.2f}"
            )
            print(
                f"Mean score (captions attended -> images unattended): {scores_df[(scores_df.train_split == TEST_CAPTIONS_ATTENDED) & (scores_df.test_split == TEST_IMAGES_UNATTENDED)].value.mean():.2f} | "
                f"Max score: {scores_df[(scores_df.train_split == TEST_CAPTIONS_ATTENDED) & (scores_df.test_split == TEST_IMAGES_UNATTENDED)].value.max():.2f}"
            )
            print(
                f"Mean score (captions attended -> images attended): {scores_df[(scores_df.train_split == TEST_CAPTIONS_ATTENDED) & (scores_df.test_split == TEST_IMAGES_ATTENDED)].value.mean():.2f} | "
                f"Max score: {scores_df[(scores_df.train_split == TEST_CAPTIONS_ATTENDED) & (scores_df.test_split == TEST_IMAGES_ATTENDED)].value.max():.2f}"
            )

            print(
                f"Mean score (images attended -> images unattended): {scores_df[(scores_df.train_split == TEST_IMAGES_ATTENDED) & (scores_df.test_split == TEST_IMAGES_UNATTENDED)].value.mean():.2f} | "
                f"Max score: {scores_df[(scores_df.train_split == TEST_IMAGES_ATTENDED) & (scores_df.test_split == TEST_IMAGES_UNATTENDED)].value.max():.2f}"
            )
            print(
                f"Mean score (images attended -> captions unattended): {scores_df[(scores_df.train_split == TEST_IMAGES_ATTENDED) & (scores_df.test_split == TEST_CAPTIONS_UNATTENDED)].value.mean():.2f} | "
                f"Max score: {scores_df[(scores_df.train_split == TEST_IMAGES_ATTENDED) & (scores_df.test_split == TEST_CAPTIONS_UNATTENDED)].value.max():.2f}"
            )
            print(
                f"Mean score (images attended -> captions attended): {scores_df[(scores_df.train_split == TEST_IMAGES_ATTENDED) & (scores_df.test_split == TEST_CAPTIONS_ATTENDED)].value.mean():.2f} | "
                f"Max score: {scores_df[(scores_df.train_split == TEST_IMAGES_ATTENDED) & (scores_df.test_split == TEST_CAPTIONS_ATTENDED)].value.max():.2f}"
            )

            results_file_path = get_results_file_path(hemi, subject, searchlight_mode_from_args(args))
            scores_df.to_csv(results_file_path)


def searchlight_mode_from_args(args):
    if args.radius is not None:
        return f"radius_{args.radius}"
    elif args.n_neighbors is not None:
        return f"n_neighbors_{args.n_neighbors}"
    else:
        raise RuntimeError("Need to set either radius or n_neighbors arg!")


def get_results_dir(hemi, subject, mode):
    results_dir = os.path.join(
        SEARCHLIGHT_ADDITIONAL_TEST_CLASSIFICATION_OUT_DIR, subject, hemi, mode
    )
    return results_dir


def get_results_file_path(hemi, subject, mode):
    results_dir = get_results_dir(hemi, subject, mode)
    return os.path.join(results_dir, f"results.csv")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS_ADDITIONAL_TEST)

    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)

    parser.add_argument("--create-null-distr", default=False, action="store_true")
    parser.add_argument("--n-permutations-per-subject", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(SEARCHLIGHT_ADDITIONAL_TEST_CLASSIFICATION_OUT_DIR, exist_ok=True)

    run(args)
