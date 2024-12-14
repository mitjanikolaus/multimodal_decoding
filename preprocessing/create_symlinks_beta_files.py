import argparse

import nibabel as nib
from glob import glob
import os

import numpy as np
from tqdm import tqdm

from utils import SUBJECTS, FMRI_BETAS_DIR

SPLITS = ['train_image', 'train_caption', 'test_caption', 'test_image', 'imagery', 'blank', 'test_trial']

SUFFIX = "*bf(1)"


def get_subdir(split_name, beta_dir):
    subdir = os.path.join(beta_dir, f"betas_{split_name}")
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    return subdir


def create_symlinks_for_beta_files(betas_dir):
    r"""
    this function makes several subdirectories and creates symbolic links
    to the corresponding beta files. it also renames the links with the coco sample id.
    """
    betas_base_dir = os.path.join(betas_dir, 'unstructured')
    beta_file_addresses = sorted(glob(os.path.join(betas_base_dir, '**', 'beta_*.nii'), recursive=True))

    betas_train_trials = []
    for beta_path in tqdm(beta_file_addresses):
        beta_file = nib.load(beta_path)
        beta_name = beta_file.header['descrip'].item().decode()
        if 'train_trial' in beta_name:
            betas_train_trials.append(beta_file)

    slink_name = os.path.join(get_subdir('splits', betas_dir), f"beta_train_trial.nii")
    print(f"averaging: {slink_name} ({len(betas_train_trials)} files)")
    averaged = np.mean([beta_file.get_fdata() for beta_file in betas_train_trials], axis=0)
    averaged_img = nib.Nifti1Image(averaged, betas_train_trials[0].affine, betas_train_trials[0].header)
    nib.save(averaged_img, slink_name)

    all_slink_names = set()
    all_beta_relative_paths = set()
    for beta_path in tqdm(beta_file_addresses):
        beta_file = nib.load(beta_path)
        beta_name = beta_file.header['descrip'].item().decode()
        for split_name in SPLITS:
            if split_name in beta_name:
                if split_name == 'blank':
                    slink_name = os.path.join(get_subdir(split_name, betas_dir), f"beta_blank.nii")
                elif split_name == 'test_trial':
                    slink_name = os.path.join(get_subdir('splits', betas_dir), f"beta_test_trial.nii")
                else:
                    stim_id = int(beta_name.split(split_name)[1].replace(SUFFIX, "").replace("_", ""))
                    slink_name = os.path.join(get_subdir(split_name, betas_dir), f"beta_{stim_id:06d}.nii")

                if slink_name in all_slink_names:
                    raise Exception(f'slink already defined: {slink_name}')
                all_slink_names.add(slink_name)
                beta_relative_path = beta_path.replace(betas_dir, '')
                if not beta_relative_path.startswith(os.sep):
                    beta_relative_path = os.sep + beta_relative_path
                beta_relative_path = f"..{beta_relative_path}"
                if beta_relative_path in all_beta_relative_paths:
                    raise Exception(f'link target already processed: {beta_relative_path}')
                all_beta_relative_paths.add(beta_relative_path)
                os.symlink(beta_relative_path, slink_name)

    print(all_slink_names)
    print(f"Created symbolic links for {len(all_slink_names)} beta files")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for subject in args.subjects:
        print(subject)
        create_symlinks_for_beta_files(os.path.join(args.betas_dir, subject))