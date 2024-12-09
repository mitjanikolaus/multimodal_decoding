import argparse

import nibabel as nib
from glob import glob
import os

import numpy as np
from tqdm import tqdm

from preprocessing.make_spm_design_job_mat_new import FMRI_BETAS_DIR
from utils import SUBJECTS

SPLITS = ['train_image', 'train_caption']
SPLITS_REPEATED = ['test_caption', 'test_image', 'imagery', 'blank']

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
    beta_file_addresses = sorted(glob(os.path.join(betas_dir, 'split_*', 'beta_*.nii'), recursive=True))

    all_slink_names = set()
    all_beta_relative_paths = set()
    for beta_path in tqdm(beta_file_addresses):
        beta_file = nib.load(beta_path)
        beta_name = beta_file.header['descrip'].item().decode()

        for split_name in SPLITS:
            if split_name in beta_name:
                if split_name == 'blank':
                    slink_name = os.path.join(get_subdir(split_name, betas_dir), f"beta_blank.nii")
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

    print(f"Created symbolic links for {len(all_slink_names)} beta files")

    for split_name in SPLITS_REPEATED:
        print(split_name)
        repeated_betas = {}
        for beta_path in tqdm(beta_file_addresses):
            beta_file = nib.load(beta_path)
            beta_name = beta_file.header['descrip'].item().decode()

            if split_name in beta_name:
                if split_name == 'blank':
                    slink_name = os.path.join(get_subdir(split_name, betas_dir), f"beta_blank.nii")
                else:
                    stim_id = int(beta_name.split(split_name)[1].replace(SUFFIX, "").replace("_", ""))
                    slink_name = os.path.join(get_subdir(split_name, betas_dir), f"beta_{stim_id:06d}.nii")
                if slink_name in repeated_betas:
                    repeated_betas[slink_name] = [beta_file]
                else:
                    repeated_betas[slink_name].append(beta_file)

        print(f"total: {len(repeated_betas)}")
        for slink_name, beta_files in tqdm(repeated_betas.items(), desc="averaging"):
            assert len(beta_files) == 2
            averaged = np.mean([beta_files[0].get_fdata(), beta_files[1].get_fdata()], axis=0)
            beta_files[0].dataobj = averaged
            nib.save(beta_files[0], slink_name)


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