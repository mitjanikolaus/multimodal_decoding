import argparse

import numpy as np
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.surface import surface

from sklearn import neighbors
import os
import pickle

from tqdm import tqdm

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS, get_nn_latent_data, \
    get_default_features, Normalize
from analyses.searchlight import pairwise_acc_captions, pairwise_acc_images, get_results_dir, \
    NUM_TEST_STIMULI

from utils import VISION_MEAN_FEAT_KEY, SURFACE_LEVEL_FMRI_DIR


DEFAULT_N_JOBS = 2


def run(args):
    for subject in args.subjects:
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

                    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
                    for hemi in args.hemis:
                        print("Hemisphere: ", hemi)
                        print(f"test_fmri_hemi shape: {test_fmri[hemi].shape}")

                        results_dir = get_results_dir(args, features, hemi, model_name, subject, training_mode)
                        estimators_file_name = f"alpha_{args.l2_regularization_alpha}_estimators.p"
                        estimators = pickle.load(open(os.path.join(results_dir, estimators_file_name), 'rb'))

                        X = test_fmri[hemi]
                        nan_locations = np.isnan(X[0])
                        assert np.all(nan_locations == np.isnan(X[-1]))
                        X = X[:, ~nan_locations]

                        infl_mesh = fsaverage[f"infl_{hemi}"]
                        coords, _ = surface.load_surf_mesh(infl_mesh)
                        coords = coords[~nan_locations]

                        nn = neighbors.NearestNeighbors(radius=args.radius)
                        if args.radius is not None:
                            adjacency = [np.argwhere(arr == 1)[:, 0] for arr in
                                         nn.fit(coords).radius_neighbors_graph(coords).toarray()]
                        elif args.n_neighbors is not None:
                            distances, adjacency = nn.fit(coords).kneighbors(coords, n_neighbors=args.n_neighbors)
                        else:
                            raise RuntimeError("Need to set either radius or n_neighbors arg!")

                        preds = [estimator.predict(X[:, adj]) for estimator, adj in zip(estimators, adjacency)]
                        preds = [Normalize(pred.mean(axis=0), pred.std(axis=0))(pred) for pred in tqdm(preds)]

                        def shuffle_and_calc_scores(latents, predictions, id, n_iters, print_interval=10):
                            results = []
                            for iter in range(n_iters):
                                np.random.shuffle(latents[:NUM_TEST_STIMULI // 2])
                                np.random.shuffle(latents[NUM_TEST_STIMULI // 2:])
                                scores = [
                                    {"test_captions": pairwise_acc_captions(latents, predictions[i], normalize=False),
                                     "test_images": pairwise_acc_images(latents, predictions[i], normalize=False)}
                                    for i in range(len(predictions))
                                ]
                                results.append(scores)
                                if iter % print_interval == 0:
                                    print(f"Thread {id}: finished {iter}/{n_iters}")

                            return results

                        n_iters_per_thread = args.n_iterations // DEFAULT_N_JOBS
                        all_scores = Parallel(n_jobs=DEFAULT_N_JOBS)(
                            delayed(shuffle_and_calc_scores)(
                                test_data_latents.copy(),
                                preds.copy(),
                                id,
                                n_iters_per_thread,
                            )
                            for id in range(DEFAULT_N_JOBS)
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
