import argparse
import os
import nibabel as nib
import numpy as np
from nilearn.image import smooth_img

from utils import SUBJECTS, FREESURFER_BASE_DIR, FMRI_RAW_DATA_DIR, FMRI_DATA_DIR, FMRI_PREPROCESSING_DATASINK_DIR


def get_graymatter_mask_path(subject, mni=False):
    file_suffix = "_mni" if mni else ""

    mask_image_path = os.path.join(
        FMRI_DATA_DIR, 'graymatter_masks', subject, f'mask{file_suffix}.nii'
    )
    return mask_image_path


# def convert_mask_to_mni(subject):
#     print('Converting mask to MNI space')
#     mask_file = get_graymatter_mask_path(subject, mni=False)
#     out_file = get_graymatter_mask_path(subject, mni=True)
#
#     reg_file = f'{FREESURFER_BASE_DIR}/regfiles/{subject}/spm2fs.change-name.lta'
#     os.makedirs(os.path.dirname(out_file), exist_ok=True)
#     conv_cmd = f'mri_vol2vol --mov "{mask_file}" --reg "{reg_file}" --o "{out_file}" --tal --talres 2 --interp nearest'
#     result_code = os.system(conv_cmd)
#     if result_code != 0:
#         raise RuntimeError(f"mri_vol2vol failed with error code {result_code}")
#     print(f"Saved MNI mask to {out_file}")
#     mask_mni = nib.load(out_file)
#     mask_mni_data = mask_mni.get_fdata()
#     print(f"MNI space gray matter mask size: {mask_mni_data.sum()} ({mask_mni_data.mean()*100:.2f}%)")


def run(args):
    for subject in args.subjects:
        print(subject)

        print('Creating mask')
        # c1_image_path = os.path.join(FMRI_RAW_DATA_DIR, 'corrected_anat', subject, f'c1{subject}_ses-01_run-01_T1W.nii')
        c1_image_path = os.path.join(FMRI_PREPROCESSING_DATASINK_DIR, 'segmented', subject, f'c1{subject}_ses-01_run-01_T1W_downsampled_2mm.nii')
        c1_img = nib.load(c1_image_path)
        c1_img_data = c1_img.get_fdata()

        # c2_image_path = os.path.join(FMRI_PREPROCESSING_DATASINK_DIR, 'segmented', subject, f'c2{subject}_ses-01_run-01_T1W_downsampled_2mm.nii')
        # c2_img = nib.load(c2_image_path)
        # c2_img_data = c2_img.get_fdata()

        # c1_img = smooth_img(c1_img, 4)
        data_masked = c1_img_data.copy()
        # data_masked[(c1_img_data > 0) | (c2_img_data > 0)] = 1
        data_masked[c1_img_data > 0] = 1
        data_masked[data_masked < 1] = 0
        data_masked = data_masked.astype(int)
        print(f"Subject-space gray matter mask size: {data_masked.sum()} ({data_masked.mean()*100:.2f}%)")

        mask_img = nib.Nifti1Image(data_masked, c1_img.affine, c1_img.header)

        mask_image_path = get_graymatter_mask_path(subject, mni=False)
        os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)

        nib.save(mask_img, mask_image_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)