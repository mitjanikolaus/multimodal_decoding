
import argparse
import os
import nibabel as nib

from utils import SUBJECTS, FREESURFER_BASE_DIR, FMRI_RAW_DATA_DIR, FMRI_PREPROCESSED_DATA_DIR


def convert_mask_to_mni(mask_file, subject, out_file):
    print('Converting mask to MNI space')
    reg_file = f'{FREESURFER_BASE_DIR}/regfiles/{subject}/spm2fs.change-name.lta'
    conv_cmd = f'mri_vol2vol --mov "{mask_file}" --reg "{reg_file}" --o "{out_file}" --tal --talres 2 --interp nearest'
    result_code = os.system(conv_cmd)
    if result_code != 0:
        raise RuntimeError(f"mri_vol2vol failed with error code {result_code}")


def get_graymatter_mask_path(subject):
    mni_mask_image_path = os.path.join(FMRI_PREPROCESSED_DATA_DIR, 'graymatter_masks', subject, f'mask.nii')
    return mni_mask_image_path


def run(args):
    os.environ["SUBJECTS_DIR"] = f"{FREESURFER_BASE_DIR}/subjects"

    for subject in args.subjects:
        print(subject)

        print('Creating mask')
        c1_image_path = os.path.join(FMRI_RAW_DATA_DIR, 'corrected_anat', subject, f'c1{subject}_ses-01_run-01_T1W.nii')
        c1_img = nib.load(c1_image_path)
        c1_img_data = c1_img.get_fdata()
        data_masked = c1_img_data.copy()
        data_masked[data_masked > 0] = 1
        mask_img = nib.Nifti1Image(data_masked, c1_img.affine, c1_img.header)

        mask_image_path = os.path.join(FMRI_PREPROCESSED_DATA_DIR, 'graymatter_masks', subject, f'mask_orig.nii')
        os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)
        nib.save(mask_img, mask_image_path)

        mni_mask_image_path = get_graymatter_mask_path(subject)
        convert_mask_to_mni(mask_image_path, subject, mni_mask_image_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
