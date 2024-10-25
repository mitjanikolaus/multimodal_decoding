import argparse
import os
from glob import glob

import nibabel as nib
import numpy as np
from nilearn.image import resample_img
from tqdm import tqdm

from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import SUBJECTS, FMRI_PREPROCESSED_DATA_DIR, nipype_subject_id

DIMS_ORIGINAL_SPACE = (170, 240, 240)
DIMS_MNI_305_2MM = (76, 76, 93)
VOXEL_SIZE_MNI_305_2MM = (2, 2, 2)
MNI_305_2MM_AFFINE = np.array([[-2, 0, 0, 74], [0, 0, 2, -109], [0, -2, 0, 85], [0, 0, 0, 1]])


def downsample_img(path, path_out, interpolation='continuous'):
    img = nib.load(path)
    assert img.shape[:3] == DIMS_ORIGINAL_SPACE

    downsampled_img = resample_img(
        img, target_shape=DIMS_MNI_305_2MM, target_affine=MNI_305_2MM_AFFINE, interpolation=interpolation
    )

    nib.save(downsampled_img, path_out)


def downsample_graymatter_masks(args):
    print("downsampling graymatter masks")
    for subject in args.subjects:
        print(subject)

        graymatter_mask_path = get_graymatter_mask_path(subject, downsampled=False)
        out_path = get_graymatter_mask_path(subject, downsampled=True)
        downsample_img(graymatter_mask_path, out_path, interpolation='nearest')


def downsample_functional_scans(args):
    print("downsampling functional scans")
    for subject in args.subjects:
        print(subject)

        img_paths = glob(
            os.path.join(
                FMRI_PREPROCESSED_DATA_DIR, 'preprocess_workflow', nipype_subject_id(subject), '*', 'coregister',
                '*bold.nii'
            )
        )
        for img_path in tqdm(img_paths):
            out_path = img_path.replace("coregister", "coregister_downsampled")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            assert out_path != img_path

            downsample_img(img_path, out_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    downsample_graymatter_masks(args)
    downsample_functional_scans(args)
