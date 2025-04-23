import argparse
import os

from preprocessing.fmri_preprocessing import DEFAULT_ANAT_SCAN_SUFFIX
from utils import FREESURFER_SUBJECTS_DIR, FMRI_RAW_DATA_DIR, SUBJECTS


def run(args):
    os.environ["SUBJECTS_DIR"] = FREESURFER_SUBJECTS_DIR
    os.makedirs(FREESURFER_SUBJECTS_DIR, exist_ok=True)

    assert os.path.isfile(args.anat_scan_path)
    cmd = f"recon-all -s {args.subject} -i {args.anat_scan_path} -all"
    os.system(cmd)


def get_args():
    parser = argparse.ArgumentParser()

    default_path = os.path.join(FMRI_RAW_DATA_DIR, 'corrected_anat', SUBJECTS[0],
                                f'{SUBJECTS[0]}_ses-01_run-01_T1W{DEFAULT_ANAT_SCAN_SUFFIX}.nii')
    parser.add_argument("--anat-scan-path", type=str, default=default_path)

    parser.add_argument("--subject", type=str, default=SUBJECTS[0])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
