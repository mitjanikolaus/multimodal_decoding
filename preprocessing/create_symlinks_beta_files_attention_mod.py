import argparse

import os

from data import IMAGERY
from preprocessing.create_symlinks_beta_files import create_symlinks_for_beta_files
from preprocessing.make_spm_design_job_mat_attention_mod import TEST_IMAGE_UNATTENDED, TEST_CAPTION_ATTENDED, \
    TEST_CAPTION_UNATTENDED, FIXATION, TEST_IMAGE_ATTENDED, FIXATION_WHITESCREEN
from utils import ATTENTION_MOD_SUBJECTS, ATTENTION_MOD_FMRI_BETAS_DIR

SPLITS = [TEST_IMAGE_ATTENDED, TEST_IMAGE_UNATTENDED, TEST_CAPTION_ATTENDED, TEST_CAPTION_UNATTENDED, IMAGERY, FIXATION,
          FIXATION_WHITESCREEN]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, nargs='+', default=ATTENTION_MOD_SUBJECTS)
    parser.add_argument("--betas-dir", type=str, default=ATTENTION_MOD_FMRI_BETAS_DIR)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for subject in args.subjects:
        print(subject)
        create_symlinks_for_beta_files(os.path.join(args.betas_dir, subject), SPLITS)
