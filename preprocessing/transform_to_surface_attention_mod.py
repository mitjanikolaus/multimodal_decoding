import argparse

import os

from data import SPLIT_IMAGERY_WEAK, \
    SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, SPLIT_TEST_IMAGE_UNATTENDED, SPLIT_TEST_CAPTION_UNATTENDED
from preprocessing.transform_to_surface import to_surface
from utils import DEFAULT_RESOLUTION, ATTENTION_MOD_FMRI_BETAS_DIR, ATTENTION_MOD_FMRI_BETAS_SURFACE_DIR, \
    ATTENTION_MOD_SUBJECTS, HEMIS


def run(args):
    for subject in args.subjects:
        print(subject)
        to_surface(subject, SPLIT_IMAGERY_WEAK, args)
        to_surface(subject, SPLIT_TEST_IMAGE_ATTENDED, args)
        to_surface(subject, SPLIT_TEST_CAPTION_ATTENDED, args)
        to_surface(subject, SPLIT_TEST_IMAGE_UNATTENDED, args)
        to_surface(subject, SPLIT_TEST_CAPTION_UNATTENDED, args)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=ATTENTION_MOD_FMRI_BETAS_DIR)
    parser.add_argument("--out-dir", type=str, default=ATTENTION_MOD_FMRI_BETAS_SURFACE_DIR)

    parser.add_argument("--subjects", type=str, nargs='+', default=ATTENTION_MOD_SUBJECTS)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--hemis", type=str, nargs="+", default=HEMIS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run(args)
