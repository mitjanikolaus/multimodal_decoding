############################################
# Training multimodal linear brain decoders
# inputs can be of any modality
# outputs are uni-modal
############################################
import argparse
import time

import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.decoding import SearchLight
from nilearn.decoding.searchlight import search_light
from nilearn.image import new_img_like, get_data
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_img
from nilearn.surface import surface
from sklearn import neighbors
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
import os
from glob import glob
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, TEST_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS, get_nn_latent_data, \
    get_default_features, pairwise_accuracy, get_run_str

from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, VISION_MEAN_FEAT_KEY

DEFAULT_N_JOBS = 20
NUM_CV_SPLITS = 3

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")


def get_graymatter_mask(subject):
    fmri_data_dir = os.path.join(TWO_STAGE_GLM_DATA_DIR, subject)
    gray_matter_mask_address = os.path.join(fmri_data_dir, f'unstructured', 'mask.nii')
    # gray_matter_mask_img = nib.load(gray_matter_mask_address)
    # gray_matter_mask_data = gray_matter_mask_img.get_fdata().astype(np.int32)
    # print(f"Gray matter mask size: {gray_matter_mask_data.sum()}")
    return gray_matter_mask_address


def get_fmri_data(subject, mode):
    imagery_scenes = IMAGERY_SCENES[subject]

    fmri_data_dir = os.path.join(TWO_STAGE_GLM_DATA_DIR, subject)
    fmri_addresses_regex = os.path.join(fmri_data_dir, f'betas_{mode}*', '*.nii')
    fmri_betas_addresses = np.array(sorted(glob(fmri_addresses_regex)))

    stim_ids = []
    stim_types = []
    for addr in fmri_betas_addresses:
        file_name = os.path.basename(addr)
        if 'I' in file_name:  # Image
            stim_id = int(file_name[file_name.find('I') + 1:-4])
            stim_types.append('image')
        elif 'C' in file_name:  # Caption
            stim_id = int(file_name[file_name.find('C') + 1:-4])
            stim_types.append('caption')
        else:  # imagery
            stim_id = int(file_name[file_name.find('.nii') - 1:-4])
            stim_id = imagery_scenes[stim_id - 1][1]
            stim_types.append('imagery')
        stim_ids.append(stim_id)

    stim_ids = np.array(stim_ids)
    stim_types = np.array(stim_types)

    return fmri_betas_addresses, stim_ids, stim_types


def run(args):
    for training_mode in args.training_modes:
        for subject in args.subjects:
            train_fmri, train_stim_ids, train_stim_types = get_fmri_data(subject, training_mode)
            if args.subset is not None:
                train_fmri = train_fmri[:args.subset]

            for testing_mode in args.testing_modes:
                test_fmri, test_stim_ids, test_stim_types = get_fmri_data(subject, testing_mode)
                fmri_data = np.concatenate((train_fmri, test_fmri))
                train_ids = list(range(len(train_fmri)))
                test_ids = list(range(len(train_fmri), len(train_fmri) + len(test_fmri)))

                gray_matter_mask = get_graymatter_mask(subject)

                for model_name in args.models:
                    model_name = model_name.lower()

                    for features in args.features:
                        if features == FEATS_SELECT_DEFAULT:
                            features = get_default_features(model_name)

                        print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                              f"MODEL: {model_name} | FEATURES: {features} | TESTING MODE: {testing_mode}")

                        train_data_latents, nn_latent_transform = get_nn_latent_data(model_name, features,
                                                                                     args.vision_features,
                                                                                     train_stim_ids,
                                                                                     subject,
                                                                                     training_mode)
                        test_data_latents, _ = get_nn_latent_data(model_name, features, args.vision_features,
                                                                  test_stim_ids,
                                                                  subject,
                                                                  testing_mode,
                                                                  nn_latent_transform=nn_latent_transform)
                        if args.subset is not None:
                            train_data_latents = train_data_latents[:args.subset]
                        latents = np.concatenate((train_data_latents, test_data_latents))

                        fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
                        for hemi in args.hemis:
                            # Average voxels 5 mm close to the 3d pial surface
                            pial_mesh = fsaverage[f"pial_{hemi}"]
                            start = time.time()
                            radius = 5.0
                            X = surface.vol_to_surf(fmri_data, pial_mesh, radius=radius, mask_img=gray_matter_mask).T
                            for i, x in enumerate(X):
                                if i == 0:
                                    print(f"nans: {np.isnan(x).sum()}")
                                nans = np.isnan(x)
                                x[nans] = 0  # TODO
                            infl_mesh = fsaverage[f"infl_{hemi}"]
                            coords, _ = surface.load_surf_mesh(infl_mesh)
                            nn = neighbors.NearestNeighbors(radius=args.radius)
                            adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()

                            model = make_pipeline(StandardScaler(), Ridge(alpha=args.l2_regularization_alpha))
                            pairwise_acc_scorer = make_scorer(pairwise_accuracy, greater_is_better=True)
                            cv = [(train_ids, test_ids)]
                            end = time.time()
                            prepr_time = int(end - start)
                            print(f"Preprocessing time: {prepr_time}s")

                            start = time.time()
                            scores = search_light(X, latents, estimator=model, A=adjacency, cv=cv, n_jobs=args.n_jobs,
                                                  scoring=pairwise_acc_scorer, verbose=3)
                            end = time.time()
                            print(f"Preprocessing time: {prepr_time}s")
                            print(f"Searchlight time: {int(end - start)}s")
                            print(f"Mean score: {scores.mean():.2f} | Max score: {scores.max():.2f}")

                            results_dir = os.path.join(SEARCHLIGHT_OUT_DIR, training_mode, model_name, features, subject,
                                                       args.resolution, hemi)
                            os.makedirs(results_dir, exist_ok=True)
                            file_name = f"alpha_{args.l2_regularization_alpha}_test_{testing_mode}.p"
                            pickle.dump(scores, open(os.path.join(results_dir, file_name), 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)
    parser.add_argument("--testing-modes", type=str, default=['test'], nargs="+", choices=TEST_MODE_CHOICES)

    parser.add_argument("--subset", type=int, default=None)

    parser.add_argument("--models", type=str, nargs='+', default=['vilt'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=VISION_MEAN_FEAT_KEY,
                        choices=VISION_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--resolution", type=str, default="fsaverage6")

    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    parser.add_argument("--l2-regularization-alpha", type=float, default=1e3)

    parser.add_argument("--radius", type=float, default=10)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(SEARCHLIGHT_OUT_DIR, exist_ok=True)

    run(args)
