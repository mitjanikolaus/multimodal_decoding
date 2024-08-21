import argparse

import nibabel as nib
from glob import glob
import os

from tqdm import tqdm

from utils import SUBJECTS, FMRI_BETAS_DIR


def create_symlinks_for_beta_files(beta_dir):
    r"""
    this function makes several subdirectories and creates symbolic links
    to the corresponding beta files. it also renames the links with the coco sample id.
    """
    beta_file_addresses = list(
        sorted(glob(os.path.join(beta_dir, f'unstructured', '**', 'beta_*.nii'), recursive=True)))
    subdirs = {
        'train_images': os.path.join(beta_dir, f'betas_train_images'),
        'test_images': os.path.join(beta_dir, f'betas_test_images'),
        'train_captions': os.path.join(beta_dir, f'betas_train_captions'),
        'test_captions': os.path.join(beta_dir, f'betas_test_captions'),
        'imagery': os.path.join(beta_dir, f'betas_imagery'),
    }

    for dir in subdirs.values():
        if os.path.exists(dir):
            raise RuntimeError(f"Output dir already exists, please remove it before running this script: ", dir)
        os.mkdir(dir)

    all_slink_names = set()
    for beta_address in tqdm(beta_file_addresses):
        beta_file = nib.load(beta_address)
        beta_name = str(beta_file.header['descrip'].astype(str))

        slink_name = None

        index = beta_name.find('blank')
        if index != -1:
            slink_name = os.path.join(subdirs['blank'], f"beta_blank.nii")

        index = beta_name.find('imagery_')
        if index != -1:
            if slink_name is not None:
                raise Exception(f'slink already defined: {slink_name}')
            endidx = beta_name.find('*bf(1)')
            slink_name = os.path.join(subdirs['imagery'], f"beta_{beta_name[index:endidx]}.nii")

        index = beta_name.find('test_image')
        if index != -1:
            if slink_name is not None:
                raise Exception(f'slink already defined: {slink_name}')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index + 11:endidx])
            slink_name = os.path.join(subdirs['test_images'], f"beta_I{stim_id:06d}.nii")

        index = beta_name.find('train_image')
        if index != -1:
            if slink_name is not None:
                raise Exception(f'slink already defined: {slink_name}')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index + 12:endidx])
            slink_name = os.path.join(subdirs['train_images'], f"beta_I{stim_id:06d}.nii")

        index = beta_name.find('test_caption')
        if index != -1:
            if slink_name is not None:
                raise Exception(f'slink already defined: {slink_name}')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index + 13:endidx])
            slink_name = os.path.join(subdirs['test_captions'], f"beta_C{stim_id:06d}.nii")

        index = beta_name.find('train_caption')
        if index != -1:
            if slink_name is not None:
                raise Exception(f'slink already defined: {slink_name}')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index + 14:endidx])
            slink_name = os.path.join(subdirs['train_captions'], f"beta_C{stim_id:06d}.nii")

        if slink_name:
            if slink_name in all_slink_names:
                raise Exception(f'slink already defined: {slink_name}')
            all_slink_names.add(slink_name)
            print(slink_name)
            os.symlink(beta_address, slink_name)

    print(f"Created symbolic links for {len(all_slink_names)} beta files")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for subject in args.subjects:
        create_symlinks_for_beta_files(os.path.join(FMRI_BETAS_DIR, subject))
