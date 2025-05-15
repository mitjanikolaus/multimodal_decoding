import argparse

import os

from data import SPLIT_IMAGERY_WEAK, SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, \
    SPLIT_TEST_IMAGE_UNATTENDED, SPLIT_TEST_CAPTION_UNATTENDED
from preprocessing.transform_to_surface import to_surface
from utils import ATTENTION_MOD_FMRI_BETAS_DIR, ATTENTION_MOD_SUBJECTS, HEMIS


def run(args):
    splits = [SPLIT_IMAGERY_WEAK, SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, SPLIT_TEST_IMAGE_UNATTENDED,
              SPLIT_TEST_CAPTION_UNATTENDED]
    to_surface(args, splits=splits)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=ATTENTION_MOD_FMRI_BETAS_DIR)

    parser.add_argument("--subjects", type=str, nargs='+', default=ATTENTION_MOD_SUBJECTS)
    parser.add_argument("--hemis", type=str, nargs="+", default=HEMIS)

    parser.add_argument("--n-jobs", type=str, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run(args)
