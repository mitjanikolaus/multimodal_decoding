import argparse
import glob

import numpy as np
from joblib import Parallel, delayed
import os
import pickle

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS, get_nn_latent_data, \
    get_default_features
from analyses.searchlight import pairwise_acc_captions, pairwise_acc_images, get_results_dir, \
    NUM_TEST_STIMULI

from utils import VISION_MEAN_FEAT_KEY, SURFACE_LEVEL_FMRI_DIR

DEFAULT_N_JOBS = 10


def run(args):
    for subject in args.subjects:
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
                        predictions_dir = os.path.join(results_dir, "test_set_predictions")
                        pred_paths = sorted(list(glob.glob(os.path.join(predictions_dir, "*.p"))))
                        print(f"Calculating null distribution for {len(pred_paths)} locations")
                        last_idx = int(os.path.basename(pred_paths[-1])[:-2])
                        assert last_idx == len(pred_paths) - 1, last_idx

                        def shuffle_and_calc_scores(latents, pred_paths, id, n_iters, print_interval=10):
                            results = []
                            for iter in range(n_iters):
                                # shuffle indices for captions and images separately
                                np.random.shuffle(latents[:NUM_TEST_STIMULI // 2])
                                np.random.shuffle(latents[NUM_TEST_STIMULI // 2:])
                                scores = []
                                for path in pred_paths:
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

                        n_iters_per_thread = args.n_iterations // DEFAULT_N_JOBS
                        all_scores = Parallel(n_jobs=DEFAULT_N_JOBS)(
                            delayed(shuffle_and_calc_scores)(
                                test_data_latents.copy(),
                                pred_paths.copy(),
                                id,
                                n_iters_per_thread,
                            )
                            for id in range(args.n_jobs)
                        )
                        all_scores = np.concatenate(all_scores)
                        results_file_name = f"alpha_{args.l2_regularization_alpha}_null_distribution.p"
                        pickle.dump(all_scores, open(os.path.join(results_dir, results_file_name), 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--models", type=str, nargs='+', default=['vilt'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
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

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
