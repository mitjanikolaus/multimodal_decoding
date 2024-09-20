import argparse
import os
from glob import glob
from tqdm import tqdm

from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import SUBJECTS, FREESURFER_BASE_DIR, FMRI_BETAS_DIR, FMRI_SURFACE_LEVEL_DIR, HEMIS_FS, FMRI_REGFILES_DIR, \
    FMRI_PREPROCESSED_DATA_DIR, nipype_subject_id

DEFAULT_RESOLUTION = "fsaverage"


def create_lta_file(subject):
    print("creating lta file")
    reg_file_path = os.path.join(FMRI_REGFILES_DIR, subject, 'spm2fs_downsampled.lta')
    os.makedirs(os.path.dirname(reg_file_path), exist_ok=True)
    vol_file_path = os.path.join(FMRI_PREPROCESSED_DATA_DIR, f'preprocess_workflow', nipype_subject_id(subject), f'_session_id_ses-01/coregister_downsampled/rameana{subject}_ses-01_task-coco_run-01_bold.nii')
    conv_cmd =  f"mri_coreg --s {subject} --mov {vol_file_path} --reg {reg_file_path}"
    print(conv_cmd)
    result_code = os.system(conv_cmd)
    assert os.path.isfile(reg_file_path), f"LTA file creation with above command failed with error code: {result_code}"
    return reg_file_path


def transform_to_surf(vol_file, output_dir, reg_file_path, resolution):
    file_name = os.path.basename(vol_file)

    for hemi in HEMIS_FS:
        out_file_name = file_name.replace('.nii', f'_{hemi}.gii')
        out_path = os.path.join(output_dir, out_file_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        conv_cmd = f'mri_vol2surf --mov "{vol_file}" --reg "{reg_file_path}" --o "{out_path}" --hemi {hemi} --trgsubject {resolution} --projfrac 0.5'
        result_code = os.system(conv_cmd)
        if result_code != 0:
            raise RuntimeError(f"mri_vol2surf failed with error code {result_code}")


def transform_graymatter_mask(subject, reg_file_path, resolution):
    mask_path = get_graymatter_mask_path(subject)
    output_dir = os.path.join(os.path.dirname(mask_path), resolution)
    transform_to_surf(mask_path, output_dir, reg_file_path, resolution)


def run(args):
    os.environ["SUBJECTS_DIR"] = f"{FREESURFER_BASE_DIR}/subjects"

    for subject in args.subjects:
        print(subject)

        reg_file_path = create_lta_file(subject)

        transform_graymatter_mask(subject, reg_file_path, args.resolution)


        base_dir = os.path.join(FMRI_BETAS_DIR, subject)
        betas_split_dirs = os.listdir(base_dir)
        if 'unstructured' in betas_split_dirs:
            betas_split_dirs.remove('unstructured')

        for split_name in betas_split_dirs:
            print("processing: ", split_name)
            beta_files = sorted(glob(os.path.join(base_dir, split_name, f'beta*.nii')))
            output_dir = os.path.join(args.output_dir, args.resolution, subject, split_name)
            os.makedirs(output_dir, exist_ok=True)

            for beta_file in tqdm(beta_files):
                transform_to_surf(beta_file, output_dir, reg_file_path, args.resolution)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--output-dir", type=str, default=FMRI_SURFACE_LEVEL_DIR)


    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)