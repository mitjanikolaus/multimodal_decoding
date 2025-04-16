import argparse

import numpy as np
from joblib import Parallel, delayed

import os

from tqdm import tqdm

from data import get_fmri_data_paths, INDICES_TEST_STIM_IMAGE, TEST_STIM_IDS, INDICES_TEST_STIM_CAPTION, IMAGERY_SCENES, \
    SPLIT_IMAGERY, SPLIT_TRAIN, SPLIT_TEST, TEST_STIM_TYPES, IMAGERY_STIMS_IDS, IMAGERY_STIMS_TYPES, \
    IMAGE, CAPTION
from utils import SUBJECTS, FMRI_BETAS_DIR, FS_HEMI_NAMES, FREESURFER_SUBJECTS_DIR


def run(args):
    os.environ["SUBJECTS_DIR"] = FREESURFER_SUBJECTS_DIR

    for subject in args.subjects:
        print("\n", subject)
        test_fmri, test_stim_ids, test_stim_types = get_fmri_data_paths(args.betas_dir, subject, SPLIT_TEST)
        imagery_fmri, imagery_stim_ids, imagery_stim_types = get_fmri_data_paths(args.betas_dir, subject, SPLIT_IMAGERY)

        assert np.all(test_stim_types[INDICES_TEST_STIM_IMAGE] == IMAGE)
        assert np.all(test_stim_types[INDICES_TEST_STIM_CAPTION] == CAPTION)
        assert np.all(test_stim_ids == TEST_STIM_IDS)
        assert np.all(test_stim_types == TEST_STIM_TYPES)
        assert np.all(imagery_stim_ids == IMAGERY_STIMS_IDS[subject])
        assert np.all(imagery_stim_types == IMAGERY_STIMS_TYPES[subject])
        assert np.all(imagery_stim_ids == [i[1] for i in IMAGERY_SCENES[subject]])

        train_fmri, _, _ = get_fmri_data_paths(args.betas_dir, subject, SPLIT_TRAIN)

        jobs = []
        for hemi in args.hemis:
            for path in tqdm(train_fmri + test_fmri + imagery_fmri):
                path_out = path.replace(args.betas_dir, os.path.join(args.betas_dir, 'surface', hemi+os.sep))
                path_out = path_out.replace('.nii', '.gii')
                assert path != path_out

                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                reg = f"--regheader {subject}" if args.overwrite_reg is None else f"--reg {args.overwrite_reg}"
                cmd = (f"mri_vol2surf --mov {path} --o {path_out} --hemi {FS_HEMI_NAMES[hemi]} --trgsubject fsaverage "
                       f"{reg} --interp nearest") #--projfrac 0.5  #--interp trilinear
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

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    parser.add_argument("--overwrite-reg", type=str)

    parser.add_argument("--n-jobs", type=str, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
