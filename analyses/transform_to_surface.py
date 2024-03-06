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
from sklearn.metrics import make_scorer
import os
from glob import glob
import pickle

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS, get_nn_latent_data, \
    get_default_features, pairwise_accuracy
from analyses.searchlight import get_fmri_data, INDICES_TEST_STIM_IMAGE, INDICES_TEST_STIM_CAPTION, IDS_TEST_STIM, \
    get_graymatter_mask

from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, VISION_MEAN_FEAT_KEY, IDS_IMAGES_TEST, DATA_DIR

OUT_DIR = os.path.join(DATA_DIR, "fmri_surface_level")


def run(args):
    for subject in args.subjects:
        train_fmri, train_stim_ids, train_stim_types = get_fmri_data(subject, "train")

        test_fmri, test_stim_ids, test_stim_types = get_fmri_data(subject, "test")

        assert np.all(test_stim_types[INDICES_TEST_STIM_IMAGE] == "image")
        assert np.all(test_stim_types[INDICES_TEST_STIM_CAPTION] == "caption")
        assert np.all(test_stim_ids == IDS_TEST_STIM)

        gray_matter_mask = get_graymatter_mask(subject)

        fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
        for hemi in args.hemis:
            print("Hemisphere: ", hemi)

            results_file_name = f"{subject}_{hemi}_train.p"

            # Average voxels 5 mm close to the 3d pial surface
            print("transforming to surface..", end=" ")
            pial_mesh = fsaverage[f"pial_{hemi}"]
            print(train_fmri.shape)
            X = surface.vol_to_surf(train_fmri[:10], pial_mesh, radius=5.0, mask_img=gray_matter_mask).T
            print("done.")
            pickle.dump(X, open(os.path.join(OUT_DIR, results_file_name), 'wb'))
            print("saved.")

            print(test_fmri.shape)
            results_file_name = f"{subject}_{hemi}_test.p"
            X = surface.vol_to_surf(test_fmri[:10], pial_mesh, radius=5.0, mask_img=gray_matter_mask).T
            pickle.dump(X, open(os.path.join(OUT_DIR, results_file_name), 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--resolution", type=str, default="fsaverage7")

    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    run(args)
