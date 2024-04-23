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

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS, get_nn_latent_data, \
    get_default_features, pairwise_accuracy, Normalize, load_latents_transform, all_pairwise_accuracy_scores, IMAGE, \
    CAPTION

from utils import VISION_MEAN_FEAT_KEY, SURFACE_LEVEL_FMRI_DIR, INDICES_TEST_STIM_CAPTION, INDICES_TEST_STIM_IMAGE, \
    IDS_TEST_STIM, NUM_TEST_STIMULI

DEFAULT_N_JOBS = 10

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")
TEST_STIM_TYPES = np.array([CAPTION] * len(INDICES_TEST_STIM_CAPTION) + [IMAGE] * len(INDICES_TEST_STIM_IMAGE))


def train_and_test(
        estimator,
        X,
        y=None,
        *,
        train_ids,
        test_ids,
        null_distr_dir=None,
        random_seeds=None,
        list_i=None,
):
    X_train = X[train_ids]
    X_test = X[test_ids]
    y_train = y[train_ids]
    y_test = y[test_ids]
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    # y_pred_normalized = Normalize(y_pred.mean(axis=0), y_pred.std(axis=0))(y_pred)

    if null_distr_dir is not None:
        scores_null_distr = []
        for seed in random_seeds:
            shuffled_indices = create_shuffled_indices(seed)
            y_test_shuffled = y_test[shuffled_indices]

            scores = all_pairwise_accuracy_scores(y_test_shuffled, y_pred, TEST_STIM_TYPES)
            scores_null_distr.append(scores)
            # scores_null_distr.append(
            #     {
            #         "test_captions": pairwise_acc_captions(y_test_shuffled, y_pred_normalized, normalize=False),
            #         "test_images": pairwise_acc_images(y_test_shuffled, y_pred_normalized, normalize=False),
            #         "test": pairwise_acc(y_test_shuffled, y_pred_normalized, normalize=False),
            #     }
            # )

        pickle.dump(scores_null_distr, open(os.path.join(null_distr_dir, f"{list_i:010d}.p"), "wb"))

    scores = all_pairwise_accuracy_scores(y_test, y_pred, TEST_STIM_TYPES)
    # scores = {
    #     "test_captions": pairwise_acc_captions(y_test, y_pred_normalized, normalize=False),
    #     "test_images": pairwise_acc_images(y_test, y_pred_normalized, normalize=False),
    #     "test": pairwise_acc(y_test, y_pred_normalized, normalize=False)
    # }

    return scores


def custom_group_iter_search_light(
        list_rows,
        list_indices,
        estimator,
        X,
        y,
        train_ids,
        test_ids,
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
                thread_id,
                len(A),
                print_interval,
                null_distr_dir,
                random_seeds.copy(),
            )
            for thread_id, list_i in enumerate(group_iter)
        )
    return np.concatenate(scores)


# def pairwise_acc_captions(latents, predictions, normalize=True):
#     return pairwise_accuracy(latents[INDICES_TEST_STIM_CAPTION], predictions[INDICES_TEST_STIM_CAPTION],
#                              normalize=normalize)
#
#
# def pairwise_acc_images(latents, predictions, normalize=True):
#     return pairwise_accuracy(latents[INDICES_TEST_STIM_IMAGE], predictions[INDICES_TEST_STIM_IMAGE],
#                              normalize=normalize)
#
#
# def pairwise_acc(latents, predictions, normalize=True):
#     return pairwise_accuracy(latents, predictions, IDS_TEST_STIM, normalize=normalize)


def create_shuffled_indices(seed):
    np.random.seed(seed)
    num_stim_one_mod = NUM_TEST_STIMULI // 2
    shuffleidx_mod_1 = np.random.choice(range(num_stim_one_mod), size=num_stim_one_mod,
                                        replace=False)
    shuffleidx_mod_2 = np.random.choice(range(num_stim_one_mod, NUM_TEST_STIMULI),
                                        size=num_stim_one_mod, replace=False)
    return np.concatenate((shuffleidx_mod_1, shuffleidx_mod_2))


def run(args):
    random_seeds = None
    if args.create_null_distr:
        random_seeds = []
        seed = 0
        for _ in range(args.n_permutations_per_subject):
            # shuffle indices for captions and images separately until all indices have changed
            shuffled_indices = create_shuffled_indices(seed)
            while any(shuffled_indices == np.arange(NUM_TEST_STIMULI)):
                seed += 1
                shuffled_indices = create_shuffled_indices(seed)
            random_seeds.append(seed)
            seed += 1

    for subject in args.subjects:
        train_fmri = {
            "left": pickle.load(
                open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_left_{args.resolution}_train.p"), 'rb')),
            "right": pickle.load(
                open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_right_{args.resolution}_train.p"), 'rb')),
        }

        test_fmri = {
            "left": pickle.load(
                open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_left_{args.resolution}_test.p"), 'rb')),
            "right": pickle.load(
                open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_right_{args.resolution}_test.p"), 'rb')),
        }

        train_stim_ids = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_ids_train.p"), 'rb'))
        train_stim_types = pickle.load(
            open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_types_train.p"), 'rb'))

        test_stim_ids = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_ids_test.p"), 'rb'))
        test_stim_types = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_types_test.p"), 'rb'))

        for training_mode in args.training_modes:
            for model_name in args.models:
                model_name = model_name.lower()

                for features in args.features:
                    if features == FEATS_SELECT_DEFAULT:
                        features = get_default_features(model_name)

                    print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                          f"MODEL: {model_name} | FEATURES: {features}")

                    train_data_latents, nn_latent_transform = get_nn_latent_data(
                        model_name, features,
                        args.vision_features,
                        train_stim_ids,
                        train_stim_types,
                        subject,
                        training_mode,
                        recompute_std_mean=args.recompute_std_mean
                    )

                    test_data_latents, _ = get_nn_latent_data(
                        model_name, features, args.vision_features,
                                                              test_stim_ids,
                                                              test_stim_types,
                                                              subject,
                                                              "test",
                                                              nn_latent_transform=nn_latent_transform
                    )
                    latents = np.concatenate((train_data_latents, test_data_latents))

                    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
                    for hemi in args.hemis:
                        print("Hemisphere: ", hemi)
                        if training_mode == "train_captions":
                            train_fmri_hemi = train_fmri[hemi][train_stim_types == 'caption']
                        elif training_mode == "train_images":
                            train_fmri_hemi = train_fmri[hemi][train_stim_types == 'image']
                        else:
                            train_fmri_hemi = train_fmri[hemi]

                        print(f"train_fmri_hemi shape: {train_fmri_hemi.shape}")
                        print(f"test_fmri_hemi shape: {test_fmri[hemi].shape}")

                        train_ids = list(range(len(train_fmri_hemi)))
                        test_ids = list(range(len(train_fmri_hemi), len(train_fmri_hemi) + len(test_fmri[hemi])))

                        X = np.concatenate((train_fmri_hemi, test_fmri[hemi]))

                        results_dir = get_results_dir(args, features, hemi, model_name, subject, training_mode)

                        results_file_name = f"alpha_{args.l2_regularization_alpha}.p"

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
                            print(f"Max distance among {args.n_neighbors} neighbors: {distances.max():.2f}mm")
                            print(f"Mean distance among {args.n_neighbors} neighbors: {distances.mean():.2f}mm")
                            print(f"Mean max distance: {distances.max(axis=1).mean():.2f}mm")
                        else:
                            raise RuntimeError("Need to set either radius or n_neighbors arg!")

                        results_dict["adjacency"] = adjacency
                        model = make_pipeline(StandardScaler(), Ridge(alpha=args.l2_regularization_alpha))
                        start = time.time()

                        null_distr_dir = None
                        if args.create_null_distr:
                            null_distr_dir = os.path.join(results_dir, "null_distr")
                            os.makedirs(null_distr_dir, exist_ok=True)

                        scores = custom_search_light(X, latents, estimator=model, A=adjacency, train_ids=train_ids,
                                                     test_ids=test_ids, n_jobs=args.n_jobs, verbose=1,
                                                     print_interval=500,
                                                     null_distr_dir=null_distr_dir,
                                                     random_seeds=random_seeds)
                        end = time.time()
                        print(f"Searchlight time: {int(end - start)}s")
                        test_scores_caps = [score["pairwise_acc_captions"] for score in scores]
                        print(
                            f"Mean score (captions): {np.mean(test_scores_caps):.2f} | Max score: {np.max(test_scores_caps):.2f}")

                        test_scores_imgs = [score["pairwise_acc_images"] for score in scores]
                        print(
                            f"Mean score (images): {np.mean(test_scores_imgs):.2f} | Max score: {np.max(test_scores_imgs):.2f}")

                        results_dict["scores"] = scores
                        pickle.dump(results_dict, open(os.path.join(results_dir, results_file_name), 'wb'))


def mode_from_args(args):
    if args.radius is not None:
        return f"radius_{args.radius}"
    else:
        return f"n_neighbors_{args.n_neighbors}"


def get_results_dir(args, features, hemi, model_name, subject, training_mode):
    results_dir = os.path.join(SEARCHLIGHT_OUT_DIR, training_mode, model_name, features,
                               subject,
                               args.resolution, hemi)
    results_dir = os.path.join(results_dir, mode_from_args(args))
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--models", type=str, nargs='+', default=['vilt'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=VISION_MEAN_FEAT_KEY,
                        choices=VISION_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--recompute-std-mean", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--resolution", type=str, default="fsaverage7")

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
