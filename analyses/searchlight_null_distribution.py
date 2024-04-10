import argparse
from glob import glob
import warnings

import numpy as np
from joblib import Parallel, delayed
import os
import pickle

from scipy import stats
from tqdm import tqdm

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS, get_nn_latent_data, \
    get_default_features
from analyses.searchlight import pairwise_acc_captions, pairwise_acc_images, get_results_dir, \
    NUM_TEST_STIMULI, SEARCHLIGHT_OUT_DIR, mode_from_args
from analyses.searchlight_results_plotting import METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, CHANCE_VALUES, \
    METRIC_MIN_DIFF_BOTH_MODALITIES, get_adj_matrices, process_scores, calc_clusters_variable_size, \
    DEFAULT_MAX_CLUSTER_DISTANCE, DEFAULT_T_VALUE_THRESHOLD

from utils import VISION_MEAN_FEAT_KEY, SURFACE_LEVEL_FMRI_DIR, HEMIS, SUBJECTS

DEFAULT_N_JOBS = 10


def create_permutation_scores(args):
    for subject in args.subjects:
        train_stim_ids = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_ids_train.p"), 'rb'))
        train_stim_types = pickle.load(
            open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_types_train.p"), 'rb'))

        test_stim_ids = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_ids_test.p"), 'rb'))
        test_stim_types = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_types_test.p"), 'rb'))

        for training_mode in args.training_modes:
            model_name = args.model.lower()

            features = args.features
            if features == FEATS_SELECT_DEFAULT:
                features = get_default_features(model_name)

            print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                  f"MODEL: {model_name} | FEATURES: {features}")

            _, nn_latent_transform = get_nn_latent_data(model_name, features,
                                                        args.vision_features,
                                                        train_stim_ids,
                                                        train_stim_types,
                                                        subject,
                                                        training_mode)
            test_data_latents, _ = get_nn_latent_data(model_name, features, args.vision_features,
                                                      test_stim_ids,
                                                      test_stim_types,
                                                      subject,
                                                      "test",
                                                      nn_latent_transform=nn_latent_transform)

            for hemi in args.hemis:
                results_dir = get_results_dir(args, features, hemi, model_name, subject, training_mode)
                results_file_name = f"alpha_{args.l2_regularization_alpha}_null_distribution.p"
                if not os.path.isfile(os.path.join(results_dir, results_file_name)):
                    predictions_dir = os.path.join(results_dir, "test_set_predictions")
                    pred_paths = sorted(list(glob(os.path.join(predictions_dir, "*.p"))))
                    print(f"Calculating permutation scores for {len(pred_paths)} locations")
                    last_idx = int(os.path.basename(pred_paths[-1])[:-2])
                    assert last_idx == len(pred_paths) - 1, last_idx

                    def shuffle_and_calc_scores(latents, pred_paths, id, n_iters, print_interval=10):
                        results = []
                        for iter in range(n_iters):
                            # shuffle indices for captions and images separately
                            np.random.shuffle(latents[:NUM_TEST_STIMULI // 2])
                            np.random.shuffle(latents[NUM_TEST_STIMULI // 2:])
                            scores = []
                            path_iterator = pred_paths
                            if id == 0:
                                path_iterator = tqdm(pred_paths)
                            for path in path_iterator:
                                preds = pickle.load(open(path, "rb"))
                                scores.append(
                                    {
                                        "test_captions": pairwise_acc_captions(latents, preds, normalize=False),
                                        "test_images": pairwise_acc_images(latents, preds, normalize=False)
                                    }
                                )
                            results.append(scores)
                            if iter % print_interval == 0:
                                print(f"Thread {id}: finished {iter}/{n_iters}")

                        return results

                    n_iters_per_thread = args.n_iterations // args.n_jobs
                    all_scores = Parallel(n_jobs=args.n_jobs)(
                        delayed(shuffle_and_calc_scores)(
                            test_data_latents.copy(),
                            pred_paths.copy(),
                            id,
                            n_iters_per_thread,
                        )
                        for id in range(args.n_jobs)
                    )
                    all_scores = np.concatenate(all_scores)
                    pickle.dump(all_scores, open(os.path.join(results_dir, results_file_name), 'wb'))


def create_null_distribution(args):
    model = args.model
    features = args.features
    if features == FEATS_SELECT_DEFAULT:
        features = get_default_features(model)

    all_scores_null_distr = []
    alpha = args.l2_regularization_alpha

    mode = mode_from_args(args)
    results_regex = os.path.join(
        SEARCHLIGHT_OUT_DIR,
        f'train/{model}/{features}/*/{args.resolution}/*/{mode}/alpha_{str(alpha)}.p'
    )
    paths_mod_agnostic = np.array(sorted(glob(results_regex)))
    paths_mod_specific_captions = np.array(sorted(glob(results_regex.replace('train/', 'train_captions/'))))
    paths_mod_specific_images = np.array(sorted(glob(results_regex.replace('train/', 'train_images/'))))
    assert len(paths_mod_agnostic) == len(paths_mod_specific_images) == len(paths_mod_specific_captions)

    adjacency_matrices = get_adj_matrices(args.resolution, args.max_cluster_distance)

    for path_agnostic, path_caps, path_imgs in zip(paths_mod_agnostic, paths_mod_specific_captions,
                                                   paths_mod_specific_images):
        print(path_agnostic)
        hemi = os.path.dirname(path_agnostic).split("/")[-2]
        subject = os.path.dirname(path_agnostic).split("/")[-4]

        results_agnostic = pickle.load(open(path_agnostic, 'rb'))
        nan_locations = results_agnostic['nan_locations']

        null_distribution_file_name = f"alpha_{str(alpha)}_null_distribution.p"
        null_distribution_agnostic = pickle.load(
            open(os.path.join(os.path.dirname(path_agnostic), null_distribution_file_name), 'rb'))

        null_distribution_images = pickle.load(
            open(os.path.join(os.path.dirname(path_imgs), null_distribution_file_name), 'rb'))

        null_distribution_captions = pickle.load(
            open(os.path.join(os.path.dirname(path_caps), null_distribution_file_name), 'rb'))

        for i, (distr, distr_caps, distr_imgs) in enumerate(zip(null_distribution_agnostic,
                                                                null_distribution_captions,
                                                                null_distribution_images)):
            if len(all_scores_null_distr) <= i:
                all_scores_null_distr.append({subj: dict() for subj in SUBJECTS})
            scores = process_scores(distr, distr_caps, distr_imgs, nan_locations)
            all_scores_null_distr[i][subject][hemi] = scores

    def calc_t_values_null_distr(per_subject_scores, n_iter=100000):
        all_t_vals = []
        for _ in tqdm(range(n_iter)):
            t_values = {hemi: dict() for hemi in HEMIS}
            for hemi in HEMIS:
                for score_name in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]:
                    random_idx = np.random.choice(len(per_subject_scores), size=len(SUBJECTS))
                    data = np.array(
                        [per_subject_scores[idx][subj][hemi][score_name] for idx, subj in
                         zip(random_idx, SUBJECTS)])
                    popmean = CHANCE_VALUES[score_name]
                    enough_data = np.isnan(data).sum(axis=0) == 0
                    t_values[hemi][score_name] = np.array([
                        stats.ttest_1samp(x, popmean=popmean, alternative="greater")[0] if ed else np.nan for x, ed
                        in zip(data.T, enough_data)]
                    )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
                        (t_values[hemi][METRIC_DIFF_CAPTIONS], t_values[hemi][METRIC_DIFF_IMAGES]),
                        axis=0)
            all_t_vals.append(t_values)
        return all_t_vals

    t_values_null_distribution_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", model, features,
                                                   args.resolution,
                                                   mode, "t_values_null_distribution.p")
    if not os.path.isfile(t_values_null_distribution_path):
        os.makedirs(os.path.dirname(t_values_null_distribution_path), exist_ok=True)
        print(f"Calculating t-values: null distribution")
        t_values_null_distribution = calc_t_values_null_distr(all_scores_null_distr)
        pickle.dump(t_values_null_distribution, open(t_values_null_distribution_path, 'wb'))
    else:
        t_values_null_distribution = pickle.load(open(t_values_null_distribution_path, 'rb'))

    clusters_null_distribution_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", model, features,
                                                   args.resolution,
                                                   mode, "clusters_null_distribution.p")
    if not os.path.isfile(clusters_null_distribution_path):
        os.makedirs(os.path.dirname(clusters_null_distribution_path), exist_ok=True)
        print(f"Calculating clusters for null distribution")
        clusters_null_distribution = [
            calc_clusters_variable_size(vals, adjacency_matrices, args.t_value_threshold) for vals in
            tqdm(t_values_null_distribution)
        ]

        pickle.dump(clusters_null_distribution, open(clusters_null_distribution_path, 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--model", type=str, default='vilt')
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=VISION_MEAN_FEAT_KEY,
                        choices=VISION_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--resolution", type=str, default="fsaverage7")

    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-iterations", type=int, default=100)

    parser.add_argument("--max-cluster-distance", type=float, default=DEFAULT_MAX_CLUSTER_DISTANCE)
    parser.add_argument("--t-value-threshold", type=float, default=DEFAULT_T_VALUE_THRESHOLD)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # create_permutation_scores(args)
    create_null_distribution(args)
