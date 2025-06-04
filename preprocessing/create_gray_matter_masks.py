import argparse
import os
import nibabel as nib

from preprocessing.fmri_preprocessing import DEFAULT_ANAT_SCAN_SUFFIX
from utils import SUBJECTS, FMRI_DATA_DIR, FMRI_PREPROCESSING_DATASINK_DIR


def get_gray_matter_mask_path(subject, mni=False):
    file_suffix = "_mni" if mni else ""

    mask_image_path = os.path.join(
        FMRI_DATA_DIR, 'graymatter_masks', subject, f'mask{file_suffix}.nii'
    )
    return mask_image_path


def run(args):
    for subject in args.subjects:
        print(subject)

        print('Creating mask')
        c1_image_path = os.path.join(FMRI_PREPROCESSING_DATASINK_DIR, 'segmented', subject,
                                     f'c1{subject}_ses-01_run-01_T1w{args.anat_scan_suffix}.nii')
        c1_img = nib.load(c1_image_path)
        c1_img_data = c1_img.get_fdata()

        data_masked = c1_img_data.copy()
        data_masked[c1_img_data > 0] = 1
        data_masked[data_masked < 1] = 0
        data_masked = data_masked.astype(int)
        print(f"Subject-space gray matter mask size: {data_masked.sum()} ({data_masked.mean() * 100:.2f}%)")

        mask_img = nib.Nifti1Image(data_masked, c1_img.affine, c1_img.header)

        mask_image_path = get_gray_matter_mask_path(subject, mni=False)
        os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)

        nib.save(mask_img, mask_image_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--anat-scan-suffix", type=str, default=DEFAULT_ANAT_SCAN_SUFFIX)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
