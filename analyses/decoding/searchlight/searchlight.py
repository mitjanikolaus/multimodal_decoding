import argparse
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.decoding.searchlight import GroupIterator
from nilearn.surface import surface

from sklearn import neighbors
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
import os
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyses.decoding.ridge_regression_decoding import FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, \
    get_latent_features, calc_all_pairwise_accuracy_scores, LANG_FEAT_COMBINATION_CHOICES, IMAGERY, TESTING_MODE, \
    ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, standardize_latents
from data import TEST_STIM_TYPES, get_fmri_surface_data, SELECT_DEFAULT, LatentFeatsConfig, create_shuffled_indices, \
    create_null_distr_seeds

from utils import SUBJECTS, DATA_DIR, \
    DEFAULT_RESOLUTION, TRAIN_MODE_CHOICES

DEFAULT_N_JOBS = 10

SEARCHLIGHT_OUT_DIR = os.path.join(DATA_DIR, "searchlight")
SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR = os.path.join(SEARCHLIGHT_OUT_DIR, "permutation_testing_results")

METRIC_AGNOSTIC = 'agnostic'


def train_and_test(
        estimator,
        X,
        y=None,
        *,
        train_ids,
        test_ids,
        imagery_ids,
        null_distr_dir=None,
        random_seeds=None,
        list_i=None,
):
    X_train = X[train_ids]
    X_test = X[test_ids]
    X_imagery = X[imagery_ids]
    y_train = y[train_ids]
    y_test = y[test_ids]
    y_imagery = y[imagery_ids]
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    y_pred_imagery = estimator.predict(X_imagery)

    if null_distr_dir is not None:
        scores_null_distr = []
        for seed in random_seeds:
            shuffled_indices = create_shuffled_indices(seed)
            y_test_shuffled = y_test[shuffled_indices]

            np.random.seed(seed)
            assert len(y_imagery) == 3
            derangements = [[1, 2, 0], [2, 0, 1]]  # possible permutations that are different from original
            shuffled_indices_imagery = derangements[np.random.choice(len(derangements))]
            y_imagery_shuffled = y_imagery[shuffled_indices_imagery]

            scores = calc_all_pairwise_accuracy_scores(y_test_shuffled, y_pred, TEST_STIM_TYPES, y_imagery_shuffled,
                                                       y_pred_imagery)
            scores_null_distr.append(scores)

        pickle.dump(scores_null_distr, open(os.path.join(null_distr_dir, f"{list_i:010d}.p"), "wb"))

    scores = calc_all_pairwise_accuracy_scores(y_test, y_pred, TEST_STIM_TYPES, y_imagery, y_pred_imagery)

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
        total,
        print_interval=500,
        null_distr_dir=None,
        random_seeds=None,
):
    results = []
    t0 = time.time()
    for (i, row), list_i in zip(enumerate(list_rows), list_indices):
        scores = train_and_test(estimator, X[:, row], y, train_ids=train_ids, test_ids=test_ids,
                                imagery_ids=imagery_ids,
                                null_distr_dir=null_distr_dir, random_seeds=random_seeds, list_i=list_i)
        results.append(scores)
        if print_interval > 0:
            if i % print_interval == 0:
                # If there is only one job, progress information is fixed
                crlf = "\r" if total == len(list_rows) else "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    f"Job #{thread_id}, processed {i}/{len(list_rows)} vertices "
                    f"({percent:0.2f}%, {round(remaining / 60)} minutes remaining){crlf}"
                )
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
        print_interval=500,
        null_distr_dir=None,
        random_seeds=None,
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
                len(A),
                print_interval,
                null_distr_dir,
                random_seeds.copy() if random_seeds is not None else None,
            )
            for thread_id, list_i in enumerate(group_iter)
        )
    return np.concatenate(scores)


def run(args):
    random_seeds = None
    if args.create_null_distr:
        random_seeds = create_null_distr_seeds(args.n_permutations_per_subject)

    for subject in args.subjects:
        for training_mode in args.training_modes:
            train_fmri, train_stim_ids, train_stim_types = get_fmri_surface_data(subject, training_mode,
                                                                                 args.resolution)
            test_fmri, test_stim_ids, test_stim_types = get_fmri_surface_data(subject, TESTING_MODE, args.resolution)
            assert np.all(test_stim_types == TEST_STIM_TYPES)
            imagery_fmri, imagery_stim_ids, imagery_stim_types = get_fmri_surface_data(subject, IMAGERY,
                                                                                       args.resolution)

            feats_config = LatentFeatsConfig(
                args.model, args.features, args.test_features, args.vision_features, args.lang_features
            )

            print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                  f"MODEL: {feats_config.model} | FEATURES: {feats_config.features}")

            train_latents = get_latent_features(feats_config.model, feats_config, train_stim_ids, train_stim_types)
            test_latents = get_latent_features(
                feats_config.model, feats_config, test_stim_ids, test_stim_types, test_mode=True
            )
            imagery_latents = get_latent_features(
                feats_config.model, feats_config, imagery_stim_ids, imagery_stim_types, test_mode=True
            )
            train_latents, test_latents, imagery_latents = standardize_latents(
                train_latents, test_latents, imagery_latents
            )

            latents = np.concatenate((train_latents, test_latents, imagery_latents))

            fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
            for hemi in args.hemis:
                print("Hemisphere: ", hemi)
                print(f"train_fmri shape: {train_fmri[hemi].shape}")
                print(f"test_fmri shape: {test_fmri[hemi].shape}")
                print(f"imagery_fmri shape: {imagery_fmri[hemi].shape}")

                train_ids = list(range(len(train_fmri[hemi])))
                test_ids = list(range(len(train_fmri[hemi]), len(train_fmri[hemi]) + len(test_fmri[hemi])))
                imagery_ids = list(range(len(train_ids) + len(test_ids),
                                         len(train_ids) + len(test_ids) + len(imagery_fmri[hemi])))

                X = np.concatenate((train_fmri[hemi], test_fmri[hemi], imagery_fmri[hemi]))

                nan_locations = np.isnan(X[0])
                assert np.all(nan_locations == np.isnan(X[-1]))
                X = X[:, ~nan_locations]

                infl_mesh = fsaverage[f"infl_{hemi}"]
                coords, _ = surface.load_surf_mesh(infl_mesh)
                coords = coords[~nan_locations]

                nn = neighbors.NearestNeighbors(radius=args.radius)
                results_dict = {}
                results_dict["nan_locations"] = nan_locations
                if args.radius is not None:
                    adjacency = [np.argwhere(arr == 1)[:, 0] for arr in
                                 nn.fit(coords).radius_neighbors_graph(coords).toarray()]
                    n_neighbors = [len(adj) for adj in adjacency]
                    results_dict["n_neighbors"] = n_neighbors
                    print(
                        f"Number of neighbors within {args.radius}mm radius: {np.mean(n_neighbors):.1f} "
                        f"(max: {np.max(n_neighbors):.0f} | min: {np.min(n_neighbors):.0f})")
                elif args.n_neighbors is not None:
                    distances, adjacency = nn.fit(coords).kneighbors(coords, n_neighbors=args.n_neighbors)
                    results_dict["distances"] = distances
                    print(f"Max radius {args.n_neighbors} neighbors: {distances.max():.2f}mm")
                    print(f"Mean radius: {distances.max(axis=1).mean():.2f}mm")
                else:
                    raise RuntimeError("Need to set either radius or n_neighbors arg!")

                results_dict["adjacency"] = adjacency
                model = make_pipeline(StandardScaler(), Ridge(alpha=args.l2_regularization_alpha))
                start = time.time()

                results_dir = get_results_dir(
                    args, feats_config.combined_feats, hemi, feats_config.model, subject, training_mode
                )
                null_distr_dir = None
                if args.create_null_distr:
                    null_distr_dir = os.path.join(results_dir, "null_distr")
                    os.makedirs(null_distr_dir, exist_ok=True)

                scores = custom_search_light(X, latents, estimator=model, A=adjacency, train_ids=train_ids,
                                             test_ids=test_ids, imagery_ids=imagery_ids, n_jobs=args.n_jobs,
                                             verbose=1,
                                             print_interval=500,
                                             null_distr_dir=null_distr_dir,
                                             random_seeds=random_seeds)
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

                results_dict["scores"] = scores
                results_file_name = f"alpha_{args.l2_regularization_alpha}.p"
                pickle.dump(results_dict, open(os.path.join(results_dir, results_file_name), 'wb'))


def mode_from_args(args):
    if args.radius is not None:
        return f"radius_{args.radius}"
    else:
        return f"n_neighbors_{args.n_neighbors}"


def get_results_dir(args, features, hemi, model_name, subject, training_mode):
    results_dir = os.path.join(SEARCHLIGHT_OUT_DIR, training_mode, model_name, features, subject, args.resolution, hemi)
    results_dir = os.path.join(results_dir, mode_from_args(args))
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

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

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

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
