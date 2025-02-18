import argparse

import numpy as np
from nilearn import datasets
from nilearn.surface import surface

import os
import pickle

from data import get_fmri_data_paths, INDICES_TEST_STIM_IMAGE, TEST_STIM_IDS, INDICES_TEST_STIM_CAPTION, IMAGERY_SCENES, \
    SPLIT_IMAGERY, SPLIT_TRAIN, SPLIT_TEST, TEST_STIM_TYPES, IMAGERY_STIMS_IDS, IMAGERY_STIMS_TYPES, SPLIT_IMAGERY_WEAK, \
    IDS_IMAGES_IMAGERY_WEAK, IDS_IMAGES_TEST_ATTENTION_MOD, SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, \
    SPLIT_TEST_IMAGE_UNATTENDED, SPLIT_TEST_CAPTION_UNATTENDED, IMAGE, CAPTION, SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS, \
    IDS_IMAGES_TEST
from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import FMRI_BETAS_SURFACE_DIR, SUBJECTS, DEFAULT_RESOLUTION, FMRI_BETAS_DIR, FMRI_STIM_INFO_DIR, HEMIS


def to_surface(subject, split, args):
    fmri, stim_ids, stim_types = get_fmri_data_paths(args.betas_dir, subject, split)
    gray_matter_mask = get_graymatter_mask_path(subject)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    if split == SPLIT_TEST_IMAGES:
        assert np.all(stim_types == IMAGE)
        assert np.all(stim_ids == IDS_IMAGES_TEST)

    if split == SPLIT_TEST_CAPTIONS:
        assert np.all(stim_types == CAPTION)
        assert np.all(stim_ids == IDS_IMAGES_TEST)

    elif split == SPLIT_IMAGERY:
        assert np.all(stim_ids == IMAGERY_STIMS_IDS[subject])
        assert np.all(stim_types == IMAGERY_STIMS_TYPES[subject])
        assert np.all(stim_ids == [i[1] for i in IMAGERY_SCENES[subject]])

    elif split == SPLIT_IMAGERY_WEAK:
        assert np.all(stim_ids == IDS_IMAGES_IMAGERY_WEAK)

    elif split in [SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, SPLIT_TEST_IMAGE_UNATTENDED,
                   SPLIT_TEST_CAPTION_UNATTENDED]:
        assert np.all(stim_ids == IDS_IMAGES_TEST_ATTENTION_MOD)

    if split == SPLIT_TRAIN:
        pickle.dump(stim_ids, open(os.path.join(FMRI_STIM_INFO_DIR, f"{subject}_stim_ids_train.p"), 'wb'))
        pickle.dump(stim_types, open(os.path.join(FMRI_STIM_INFO_DIR, f"{subject}_stim_types_train.p"), 'wb'))

        for hemi in args.hemis:
            print("Hemisphere: ", hemi)
            pial_mesh = fsaverage[f"pial_{hemi}"]

            print("transforming to surface.. (train part 1)", end=" ")
            X = surface.vol_to_surf(fmri[:2500], pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name_1 = f"{subject}_{hemi}_train_1.p"
            pickle.dump(X, open(os.path.join(args.out_dir, results_file_name_1), 'wb'))
            print("saved.")

            print("transforming to surface.. (train part 2)", end=" ")
            pial_mesh = fsaverage[f"pial_{hemi}"]
            X = surface.vol_to_surf(fmri[2500:5000], pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name_2 = f"{subject}_{hemi}_train_2.p"
            pickle.dump(X, open(os.path.join(args.out_dir, results_file_name_2), 'wb'))
            print("saved.")

            print("transforming to surface.. (train part 3)", end=" ")
            pial_mesh = fsaverage[f"pial_{hemi}"]
            X = surface.vol_to_surf(fmri[5000:7500], pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name_3 = f"{subject}_{hemi}_train_3.p"
            pickle.dump(X, open(os.path.join(args.out_dir, results_file_name_3), 'wb'))
            print("saved.")

            print("transforming to surface.. (train part 4)", end=" ")
            pial_mesh = fsaverage[f"pial_{hemi}"]
            X = surface.vol_to_surf(fmri[7500:], pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name_4 = f"{subject}_{hemi}_train_4.p"
            pickle.dump(X, open(os.path.join(args.out_dir, results_file_name_4), 'wb'))
            print("saved.")

            X_1 = pickle.load(open(os.path.join(args.out_dir, results_file_name_1), 'rb'))
            X_2 = pickle.load(open(os.path.join(args.out_dir, results_file_name_2), 'rb'))
            X_3 = pickle.load(open(os.path.join(args.out_dir, results_file_name_3), 'rb'))
            X_4 = pickle.load(open(os.path.join(args.out_dir, results_file_name_4), 'rb'))
            results_file_name = f"{subject}_{hemi}_{args.resolution}_train.p"
            pickle.dump(np.concatenate((X_1, X_2, X_3, X_4)), open(os.path.join(args.out_dir, results_file_name), 'wb'))
            os.remove(os.path.join(args.out_dir, results_file_name_1))
            os.remove(os.path.join(args.out_dir, results_file_name_2))
            os.remove(os.path.join(args.out_dir, results_file_name_3))
            os.remove(os.path.join(args.out_dir, results_file_name_4))

    else:
        for hemi in args.hemis:
            print("Hemisphere: ", hemi)
            pial_mesh = fsaverage[f"pial_{hemi}"]

            print(f"transforming to surface.. ({split})", end=" ")
            X = surface.vol_to_surf(fmri, pial_mesh, mask_img=gray_matter_mask).T
            print("done.")
            results_file_name = f"{subject}_{hemi}_{args.resolution}_{split}.p"
            pickle.dump(X, open(os.path.join(args.out_dir, results_file_name), 'wb'))
            print("saved.")


def run(args):
    for subject in args.subjects:
        print("\n", subject)
        # to_surface(subject, SPLIT_TRAIN, args)
        to_surface(subject, SPLIT_TEST_IMAGES, args)
        to_surface(subject, SPLIT_TEST_CAPTIONS, args)
        # to_surface(subject, SPLIT_IMAGERY, args)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)
    parser.add_argument("--out-dir", type=str, default=FMRI_BETAS_SURFACE_DIR)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--hemis", type=str, nargs="+", default=HEMIS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run(args)
