import argparse

import numpy as np
from nilearn import datasets
from nilearn.surface import surface

import os
from glob import glob
import pickle

from analyses.ridge_regression_decoding import DEFAULT_SUBJECTS
from analyses.searchlight import INDICES_TEST_STIM_IMAGE, INDICES_TEST_STIM_CAPTION
from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, FMRI_SURFACE_LEVEL_DIR, IDS_TEST_STIM


def get_graymatter_mask(subject):
    fmri_data_dir = os.path.join(TWO_STAGE_GLM_DATA_DIR, subject)
    gray_matter_mask_address = os.path.join(fmri_data_dir, f'unstructured', 'mask.nii')
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
    for subject in args.subjects:
        print("\n", subject)
        train_fmri, train_stim_ids, train_stim_types = get_fmri_data(subject, "train")
        test_fmri, test_stim_ids, test_stim_types = get_fmri_data(subject, "test")

        pickle.dump(train_stim_ids, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_stim_ids_train.p"), 'wb'))
        pickle.dump(train_stim_types, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_stim_types_train.p"), 'wb'))
        pickle.dump(test_stim_ids, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_stim_ids_test.p"), 'wb'))
        pickle.dump(test_stim_types, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_stim_types_test.p"), 'wb'))

        assert np.all(test_stim_types[INDICES_TEST_STIM_IMAGE] == "image")
        assert np.all(test_stim_types[INDICES_TEST_STIM_CAPTION] == "caption")
        assert np.all(test_stim_ids == IDS_TEST_STIM)

        gray_matter_mask = get_graymatter_mask(subject)

        fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
        for hemi in args.hemis:
            print("Hemisphere: ", hemi)

            print("transforming to surface.. (train part 1)", end=" ")
            pial_mesh = fsaverage[f"pial_{hemi}"]
            X = surface.vol_to_surf(train_fmri[:2500], pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name_1 = f"{subject}_{hemi}_train_1.p"
            pickle.dump(X, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_1), 'wb'))
            print("saved.")

            print("transforming to surface.. (train part 2)", end=" ")
            pial_mesh = fsaverage[f"pial_{hemi}"]
            X = surface.vol_to_surf(train_fmri[2500:5000], pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name_2 = f"{subject}_{hemi}_train_2.p"
            pickle.dump(X, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_2), 'wb'))
            print("saved.")

            print("transforming to surface.. (train part 3)", end=" ")
            pial_mesh = fsaverage[f"pial_{hemi}"]
            X = surface.vol_to_surf(train_fmri[5000:7500], pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name_3 = f"{subject}_{hemi}_train_3.p"
            pickle.dump(X, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_3), 'wb'))
            print("saved.")

            print("transforming to surface.. (train part 4)", end=" ")
            pial_mesh = fsaverage[f"pial_{hemi}"]
            X = surface.vol_to_surf(train_fmri[7500:], pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name_4 = f"{subject}_{hemi}_train_4.p"
            pickle.dump(X, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_4), 'wb'))
            print("saved.")

            X_1 = pickle.load(open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_1), 'rb'))
            X_2 = pickle.load(open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_2), 'rb'))
            X_3 = pickle.load(open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_3), 'rb'))
            X_4 = pickle.load(open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_4), 'rb'))
            results_file_name = f"{subject}_{hemi}_{args.resolution}_train.p"
            pickle.dump(np.concatenate((X_1, X_2, X_3, X_4)), open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name), 'wb'))
            os.remove(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_1))
            os.remove(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_2))
            os.remove(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_3))
            os.remove(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name_4))

            print("transforming to surface.. (test)", end=" ")
            X = surface.vol_to_surf(test_fmri, pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name = f"{subject}_{hemi}_{args.resolution}_test.p"
            pickle.dump(X, open(os.path.join(FMRI_SURFACE_LEVEL_DIR, results_file_name), 'wb'))
            print("saved.")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--resolution", type=str, default="fsaverage7")

    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(FMRI_SURFACE_LEVEL_DIR, exist_ok=True)

    run(args)
