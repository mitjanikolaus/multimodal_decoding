import argparse
import re

import nibabel as nib
from glob import glob
import os

from tqdm import tqdm

from utils import SUBJECTS, FMRI_BETAS_DIR

SPLITS = ['train_image', 'train_caption', 'test_caption', 'test_image', 'imagery', 'blank']

SUFFIX = "*bf(1)"


def get_subdir(split_name, beta_dir):
    subdir = os.path.join(beta_dir, f"betas_{split_name}")
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    return subdir


def create_symlinks_for_beta_files(betas_dir, splits):
    r"""
    this function makes several subdirectories and creates symbolic links
    to the corresponding beta files. it also renames the links with the coco sample id.
    """
    beta_file_addresses = sorted(glob(os.path.join(betas_dir, 'unstructured', 'beta_*.nii'), recursive=True))

    all_slink_names = set()
    all_beta_relative_paths = set()
    for beta_path in tqdm(beta_file_addresses):
        beta_file = nib.load(beta_path)
        beta_name = beta_file.header['descrip'].item().decode()
        beta_name = beta_name.split(' ')[-1].replace(SUFFIX, "")
        split_name = re.sub('\d', '', beta_name)
        if split_name.endswith('_'):
            split_name = split_name[:-1]

        if split_name in splits:
            if split_name in ['blank', 'fixation', 'fixation_whitescreen']:
                slink_name = os.path.join(get_subdir(split_name, betas_dir), f"beta_{split_name}.nii")
            else:
                stim_id = int(beta_name.split(split_name)[1].replace("_", ""))
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for subject in args.subjects:
        print(subject)
        create_symlinks_for_beta_files(os.path.join(args.betas_dir, subject), SPLITS)