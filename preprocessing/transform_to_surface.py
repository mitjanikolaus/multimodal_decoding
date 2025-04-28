import argparse
import shutil

import numpy as np
from joblib import Parallel, delayed

import os

from tqdm import tqdm

from data import get_fmri_data_paths, INDICES_TEST_STIM_IMAGE, TEST_STIM_IDS, INDICES_TEST_STIM_CAPTION, \
    IMAGERY_SCENES, SPLIT_IMAGERY, SPLIT_TRAIN, SPLIT_TEST, TEST_STIM_TYPES, IMAGERY_STIMS_IDS, IMAGERY_STIMS_TYPES, \
    IMAGE, CAPTION
from utils import SUBJECTS, FMRI_BETAS_DIR, FS_HEMI_NAMES, FREESURFER_SUBJECTS_DIR, HEMIS
import nibabel as nib


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

        paths = []
        for path in train_fmri + test_fmri + imagery_fmri:
            paths.append(path)

        def transform(path, betas_dir, silent=True):
            # fist transform NaNs to zeros
            img = nib.load(path)
            img_data = img.get_fdata()
            img_data[np.isnan(img_data)] = 0
            img_zeroed = nib.Nifti1Image(img_data, img.affine, img.header)
            path_img_zeroed = path.replace(betas_dir, os.path.join(betas_dir, 'nan_to_zero' + os.sep))
            assert path != path_img_zeroed
            os.makedirs(os.path.dirname(path_img_zeroed), exist_ok=True)
            nib.save(img_zeroed, path_img_zeroed)

            # then transform to surface using freesurfer
            for hemi in HEMIS:
                path_out = path.replace(betas_dir, os.path.join(betas_dir, 'surface', hemi + os.sep))
                path_out = path_out.replace('.nii', '.gii')
                assert path != path_out
                os.makedirs(os.path.dirname(path_out), exist_ok=True)

                reg = f"--regheader {subject}"
                cmd = (
                    f"mri_vol2surf --mov {path_img_zeroed} --o {path_out} --hemi {FS_HEMI_NAMES[hemi]} "
                    f"--trgsubject fsaverage {reg} --interp trilinear --projfrac-avg 0 1 0.2"
                )

                if silent:
                    cmd = cmd + " >/dev/null 2>&1"
                result_code = os.system(cmd)
                if result_code != 0:
                    raise RuntimeError(f"failed to convert {path} to surface {result_code}")

        Parallel(n_jobs=int(args.n_jobs))(
            delayed(transform)(path, args.betas_dir)
            for path in tqdm(paths)
        )

        shutil.rmtree(os.path.join(args.betas_dir, 'nan_to_zero', subject))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--n-jobs", type=str, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
