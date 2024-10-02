import argparse

import nibabel as nib
from glob import glob
import os

from tqdm import tqdm

from utils import SUBJECTS, FMRI_BETAS_DIR

SPLITS = ['train_images', 'train_captions', 'test_captions', 'test_images', 'imagery', 'blank', 'pilot_image',
          'pilot_caption', 'pilot_filler_image', 'pilot_filler_caption']

SUFFIX = "*bf(1)"


def get_subdir(split_name, beta_dir):
    subdir = os.path.join(beta_dir, f"betas_{split_name}")
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
    return subdir


def create_symlinks_for_beta_files(beta_dir):
    r"""
    this function makes several subdirectories and creates symbolic links
    to the corresponding beta files. it also renames the links with the coco sample id.
    """
    beta_base_dir = os.path.join(beta_dir, 'unstructured')
    beta_file_addresses = sorted(glob(os.path.join(beta_base_dir, '**', 'beta_*.nii'), recursive=True))

    all_slink_names = set()
    for beta_path in tqdm(beta_file_addresses):
        beta_file = nib.load(beta_path)
        beta_name = beta_file.header['descrip'].item().decode()

        for split_name in SPLITS:
            if split_name in beta_name:
                if split_name == 'blank':
                    slink_name = os.path.join(get_subdir(split_name, beta_dir), f"beta_blank.nii")
                else:
                    stim_id = int(beta_name.split(split_name)[1].replace(SUFFIX, "").replace("_", ""))
                    prefix = 'I' if 'image' in split_name else 'C' if 'caption' in split_name else ''
                    slink_name = os.path.join(get_subdir(split_name, beta_dir), f"beta_{prefix}{stim_id:06d}.nii")

                if slink_name in all_slink_names:
                    raise Exception(f'slink already defined: {slink_name}')
                all_slink_names.add(slink_name)
                beta_relative_path = beta_path.replace(beta_dir, '')
                if not beta_relative_path.startswith(os.sep):
                    beta_relative_path = os.sep + beta_relative_path
                beta_relative_path = f"..{beta_relative_path}"
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
        create_symlinks_for_beta_files(os.path.join(args.betas_dir, subject))
