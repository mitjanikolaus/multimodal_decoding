import argparse
import os
from glob import glob

from tqdm import tqdm

from utils import SUBJECTS, FREESURFER_BASE_DIR, FMRI_BETAS_DIR, FMRI_SURFACE_LEVEL_DIR, HEMIS_FS, FMRI_REGFILES_DIR, \
    FMRI_PREPROCESSED_DATA_DIR, nipype_subject_id


def create_lta_file(subject):
    print("creating lta file")
    reg_file_path = os.path.join(FMRI_REGFILES_DIR, subject, 'spm2fs')
    os.makedirs(os.path.dirname(reg_file_path), exist_ok=True)
    vol_file_path = os.path.join(FMRI_PREPROCESSED_DATA_DIR, f'preprocess_workflow', nipype_subject_id(subject), '_session_id_ses-01/coregister_downsampled/rameanasub-01_ses-01_task-coco_run-01_bold.nii')
    conv_cmd = f'tkregisterfv --mov {vol_file_path} --s {subject} --regheader --reg {reg_file_path}'
    print(conv_cmd)
    result_code = os.system(conv_cmd)
    suffix = '.change-name.lta'
    out_path = reg_file_path + suffix
    assert os.path.isfile(out_path), f"LTA file creation with above command failed with error code: {result_code}"
    return out_path


def run(args):
    os.environ["SUBJECTS_DIR"] = f"{FREESURFER_BASE_DIR}/subjects"

    for subject in args.subjects:
        print(subject)

        reg_file_path = create_lta_file(subject)

        out_dir = os.path.join(args.output_dir, subject)

        os.makedirs(out_dir, exist_ok=True)

        base_output_dir = os.path.join(args.output_dir, subject)

        base_dir = os.path.join(FMRI_BETAS_DIR, subject)
        betas_split_dirs = os.listdir(base_dir)
        if 'unstructured' in betas_split_dirs:
            betas_split_dirs.remove('unstructured')

        for split_name in betas_split_dirs:
            print("processing: ", split_name)
            os.makedirs(os.path.join(base_output_dir, split_name), exist_ok=True)

            beta_files = sorted(glob(os.path.join(base_dir, split_name, f'beta*.nii')))
            for beta_file in tqdm(beta_files):
                file_name = os.path.basename(beta_file)
                out_file_name = file_name.replace('.nii', '.gii')
                out_vol = os.path.join(base_output_dir, split_name, out_file_name)

                for hemi in HEMIS_FS:
                    conv_cmd = f'mri_vol2surf --mov "{beta_file}" --reg "{reg_file_path}" --o "{out_vol}" --hemi {hemi} --trgsubject fsaverage'
                    result_code = os.system(conv_cmd)
                    if result_code != 0:
                        raise RuntimeError(f"mri_vol2surf failed with error code {result_code}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--output-dir", type=str, default=FMRI_SURFACE_LEVEL_DIR)


    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)