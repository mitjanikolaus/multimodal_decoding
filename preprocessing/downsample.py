import argparse
from glob import glob

import numpy as np
from joblib import Parallel, delayed

import os

from tqdm import tqdm

from data import get_fmri_data_paths, INDICES_TEST_STIM_IMAGE, TEST_STIM_IDS, INDICES_TEST_STIM_CAPTION, IMAGERY_SCENES, \
    SPLIT_IMAGERY, SPLIT_TRAIN, SPLIT_TEST, TEST_STIM_TYPES, IMAGERY_STIMS_IDS, IMAGERY_STIMS_TYPES, \
    IMAGE, CAPTION
from utils import SUBJECTS, FMRI_BETAS_DIR, FS_HEMI_NAMES, FREESURFER_SUBJECTS_DIR, FMRI_PREPROCESSING_DATASINK_DIR


def run(args):
    os.environ["FSLOUTPUTTYPE"] = 'NIFTI'

    for subject in args.subjects:
        print("\n", subject)

        paths = sorted(glob(os.path.join(FMRI_PREPROCESSING_DATASINK_DIR, "coregistered", subject, '*', '*.nii')))
        print(f'downsampling {len(paths)} files')

        jobs = []
        for hemi in args.hemis:
            for path in tqdm(paths):
                path_out = path.replace("coregistered", "coregistered_downsampled")
                assert path != path_out

                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                cmd = f"flirt.fsl -in {path} -ref {path} -applyisoxfm 2.0 -nosearch -out {path_out}"
                jobs.append(cmd)

        def exec_command(cmd, silent=True):
            if silent:
                cmd = cmd + " >/dev/null 2>&1"
            result_code = os.system(cmd)
            if result_code != 0:
                raise RuntimeError(f"failed to convert {path} to surface {result_code}")

        Parallel(n_jobs=int(args.n_jobs))(
            delayed(exec_command)(cmd)
            for cmd in tqdm(jobs)
        )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--n-jobs", type=str, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
