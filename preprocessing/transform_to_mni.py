#########################################################
# Transform coregistered scans from the subject space to MNI305 space
#########################################################
import argparse
import os
from glob import glob

from tqdm import tqdm

from utils import SUBJECTS, FREESURFER_BASE_DIR, FMRI_PREPROCESSED_DATA_DIR, FMRI_PREPROCESSED_MNI_DATA_DIR


def run(args):
    os.environ["SUBJECTS_DIR"] = f"{FREESURFER_BASE_DIR}/subjects"

    for subject in args.subjects:
        print(subject)
        nipype_subject_name = f'_subject_id_{subject}'

        vol_dir = f'{FMRI_PREPROCESSED_DATA_DIR}/preprocess_workflow/{nipype_subject_name}'
        reg_file = f'{FREESURFER_BASE_DIR}/regfiles/{subject}/spm2fs.change-name.lta'
        out_dir = f'{FMRI_PREPROCESSED_MNI_DATA_DIR}/{subject}'

        os.makedirs(out_dir, exist_ok=True)

        all_vol_files = sorted(glob(os.path.join(vol_dir, '_session_*', 'coregister', f'rara{subject}_*_bold.nii')))
        for vol_file in tqdm(all_vol_files):
            idx = vol_file.find('ses-')
            sess_id = vol_file[idx:idx + 6]
            out_sess_dir = os.path.join(out_dir, sess_id)
            file_name = os.path.basename(vol_file)
            out_vol = os.path.join(out_sess_dir, file_name)
            os.makedirs(out_sess_dir, exist_ok=True)

            conv_cmd = f'mri_vol2vol --mov "{vol_file}" --reg "{reg_file}" --o "{out_vol}" --tal --talres 2'
            result_code = os.system(conv_cmd)
            if result_code != 0:
                raise RuntimeError(f"mri_vol2vol failed with error code {result_code}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
