import argparse
import sys
import time
import warnings
from glob import glob

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
from tqdm import tqdm

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, get_nn_latent_data, \
    get_default_features, calc_all_pairwise_accuracy_scores, IMAGE, \
    CAPTION, get_default_vision_features, LANG_FEAT_COMBINATION_CHOICES, get_default_lang_features, \
    get_fmri_surface_data, IMAGERY, TESTING_MODE, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, ACC_CAPTIONS, ACC_IMAGES, \
    ACC_MODALITY_AGNOSTIC

from utils import INDICES_TEST_STIM_CAPTION, INDICES_TEST_STIM_IMAGE, NUM_TEST_STIMULI, SUBJECTS, \
    correlation_num_voxels_acc, HEMIS, export_to_gifti, FS_HEMI_NAMES

DEFAULT_N_JOBS = 10

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")
TEST_STIM_TYPES = np.array([CAPTION] * len(INDICES_TEST_STIM_CAPTION) + [IMAGE] * len(INDICES_TEST_STIM_IMAGE))

BASE_METRICS = [ACC_CAPTIONS, ACC_IMAGES, ACC_MODALITY_AGNOSTIC, ACC_IMAGERY]


METRIC_CAPTIONS = 'captions'
METRIC_IMAGES = 'images'
METRIC_AGNOSTIC = 'agnostic'
METRIC_DIFF_CAPTIONS = 'captions_agno - captions_specific'
METRIC_DIFF_IMAGES = 'imgs_agno - imgs_specific'
METRIC_MIN_DIFF_BOTH_MODALITIES = 'min(captions_agno - captions_specific, imgs_agno - imgs_specific)'
METRIC_MIN = 'min_alternative'
METRIC_IMAGERY = 'imagery'
METRIC_IMAGERY_WHOLE_TEST = 'imagery_whole_test'


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

            scores = calc_all_pairwise_accuracy_scores(y_test_shuffled, y_pred, TEST_STIM_TYPES)
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
        scores = train_and_test(estimator, X[:, row], y, train_ids=train_ids, test_ids=test_ids, imagery_ids=imagery_ids,
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
        for training_mode in args.training_modes:
            train_fmri, train_stim_ids, train_stim_types = get_fmri_surface_data(subject, training_mode,
                                                                                 args.resolution)
            test_fmri, test_stim_ids, test_stim_types = get_fmri_surface_data(subject, TESTING_MODE, args.resolution)
            imagery_fmri, imagery_stim_ids, imagery_stim_types = get_fmri_surface_data(subject, IMAGERY,
                                                                                       args.resolution)

            model_name = args.model.lower()

            print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                  f"MODEL: {model_name} | FEATURES: {args.features}")

            train_data_latents, nn_latent_transform = get_nn_latent_data(
                model_name, args.features,
                args.vision_features,
                args.lang_features,
                train_stim_ids,
                train_stim_types,
                subject,
                training_mode,
            )

            test_data_latents, _ = get_nn_latent_data(
                model_name,
                args.features,
                args.vision_features,
                args.lang_features,
                test_stim_ids,
                test_stim_types,
                subject,
                TESTING_MODE,
                nn_latent_transform=nn_latent_transform
            )

            imagery_data_latents, _ = get_nn_latent_data(
                model_name,
                args.features,
                args.vision_features,
                args.lang_features,
                imagery_stim_ids,
                imagery_stim_types,
                subject,
                IMAGERY,
                nn_latent_transform=nn_latent_transform)
            latents = np.concatenate((train_data_latents, test_data_latents, imagery_data_latents))

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

                results_dir = get_results_dir(args, args.features, hemi, model_name, subject, training_mode)

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
                                             test_ids=test_ids, imagery_ids=imagery_ids, n_jobs=args.n_jobs,
                                             verbose=1,
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

                imagery_scores = [score[ACC_IMAGERY] for score in scores]
                print(
                    f"Mean score ({ACC_IMAGERY}): {np.mean(imagery_scores):.2f} | Max score: {np.max(imagery_scores):.2f}")

                imagery_whole_test_set_scores = [score[ACC_IMAGERY_WHOLE_TEST] for score in scores]
                print(
                    f"Mean score ({ACC_IMAGERY_WHOLE_TEST}): {np.mean(imagery_whole_test_set_scores):.2f} | Max score: {np.max(imagery_whole_test_set_scores):.2f}")

                results_dict["scores"] = scores
                pickle.dump(results_dict, open(os.path.join(results_dir, results_file_name), 'wb'))


def process_scores(scores_agnostic, scores_captions, scores_images, nan_locations, subj, hemi, args, n_neighbors=None):
    scores = dict()

    for metric in BASE_METRICS:
        score_name = metric.split("_")[-1]
        scores[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores[score_name][~nan_locations] = np.array([score[metric] for score in scores_agnostic])

    if "plot_n_neighbors_correlation_graph" in args and args.plot_n_neighbors_correlation_graph and (n_neighbors is not None) and (subj is not None):
        correlation_num_voxels_acc(scores, nan_locations, n_neighbors, subj, hemi)

    scores_specific_captions = dict()
    for metric in BASE_METRICS:
        score_name = metric.split("_")[-1]
        scores_specific_captions[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores_specific_captions[score_name][~nan_locations] = np.array(
            [score[metric] for score in scores_captions])

    scores_specific_images = dict()
    for metric in BASE_METRICS:
        score_name = metric.split("_")[-1]
        scores_specific_images[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores_specific_images[score_name][~nan_locations] = np.array(
            [score[metric] for score in scores_images])

    scores[METRIC_IMAGERY_WHOLE_TEST] = np.repeat(np.nan, nan_locations.shape)
    scores[METRIC_IMAGERY_WHOLE_TEST][~nan_locations] = np.array([score[ACC_IMAGERY_WHOLE_TEST] for score in scores_agnostic])

    scores[METRIC_DIFF_IMAGES] = np.array(
        [ai - si for ai, ac, si, sc in
         zip(scores[METRIC_IMAGES],
             scores[METRIC_CAPTIONS],
             scores_specific_images[METRIC_IMAGES],
             scores_specific_captions[METRIC_CAPTIONS])]
    )
    scores[METRIC_DIFF_CAPTIONS] = np.array(
        [ac - sc for ai, ac, si, sc in
         zip(scores[METRIC_IMAGES],
             scores[METRIC_CAPTIONS],
             scores_specific_images[METRIC_IMAGES],
             scores_specific_captions[METRIC_CAPTIONS])]
    )

    return scores


def load_per_subject_scores(args):
    per_subject_scores = {subj: dict() for subj in SUBJECTS}

    results_regex = os.path.join(
        SEARCHLIGHT_OUT_DIR,
        f'train/{args.model}/{args.features}/*/{args.resolution}/*/{args.mode}/alpha_{str(args.l2_regularization_alpha)}.p'
    )
    paths_mod_agnostic = np.array(sorted(glob(results_regex)))
    paths_mod_specific_captions = np.array(sorted(glob(results_regex.replace('train/', 'train_captions/'))))
    paths_mod_specific_images = np.array(sorted(glob(results_regex.replace('train/', 'train_images/'))))
    assert len(paths_mod_agnostic) == len(paths_mod_specific_images) == len(paths_mod_specific_captions)

    for path_agnostic, path_caps, path_imgs in tqdm(zip(paths_mod_agnostic, paths_mod_specific_captions,
                                                        paths_mod_specific_images), total=len(paths_mod_agnostic)):
        hemi = os.path.dirname(path_agnostic).split("/")[-2]
        subject = os.path.dirname(path_agnostic).split("/")[-4]

        results_agnostic = pickle.load(open(path_agnostic, 'rb'))
        scores_agnostic = results_agnostic['scores']
        scores_captions = pickle.load(open(path_caps, 'rb'))['scores']
        scores_images = pickle.load(open(path_imgs, 'rb'))['scores']

        nan_locations = results_agnostic['nan_locations']
        n_neighbors = results_agnostic['n_neighbors'] if 'n_neighbors' in results_agnostic else None
        scores = process_scores(
            scores_agnostic, scores_captions, scores_images, nan_locations, subject, hemi, args, n_neighbors
        )
        # print({n: round(np.nanmean(score), 4) for n, score in scores.items()})
        # print({f"{n}_max": round(np.nanmax(score), 2) for n, score in scores.items()})
        # print("")

        per_subject_scores[subject][hemi] = scores
    return per_subject_scores


def create_gifti_results_maps(args):
    args.mode = mode_from_args(args)
    per_subject_scores = load_per_subject_scores(args)

    METRICS = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_AGNOSTIC, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES,
               METRIC_IMAGERY, METRIC_IMAGERY_WHOLE_TEST]

    results_dir = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
                               args.resolution, args.mode, "acc_scores_gifti")
    os.makedirs(results_dir, exist_ok=True)

    for metric in METRICS:
        for hemi in HEMIS:
            score_hemi_avgd = np.nanmean([per_subject_scores[subj][hemi][metric] for subj in SUBJECTS], axis=0)
            path_out = os.path.join(results_dir,  f"{metric.replace(' ', '')}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(score_hemi_avgd, path_out)


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

    parser.add_argument("--model", type=str, default="blip2")
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

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

    model_name = args.model.lower()
    if args.features == FEATS_SELECT_DEFAULT:
        args.features = get_default_features(model_name)
    if args.vision_features == FEATS_SELECT_DEFAULT:
        args.vision_features = get_default_vision_features(model_name)
    if args.lang_features == FEATS_SELECT_DEFAULT:
        args.lang_features = get_default_lang_features(model_name)

    run(args)
    create_gifti_results_maps(args)

