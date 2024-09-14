import argparse
import os
from glob import glob

import nibabel as nib
from nibabel import affines
from nilearn.image import resample_img
from tqdm import tqdm

from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import SUBJECTS, FMRI_PREPROCESSED_DATA_DIR, nipype_subject_id

DIMS_ORIGINAL_SPACE = (170, 240, 240)
DIMS_MNI_305_2MM = (76, 76, 93)
VOXEL_SIZE_MNI_305_2MM = (2, 2, 2)


def downsample_graymatter_masks(args):
    print("downsampling graymatter masks")
    for subject in args.subjects:
        print(subject)

        graymatter_mask_path = get_graymatter_mask_path(subject, downsampled=False)
        mask = nib.load(graymatter_mask_path)
        assert mask.shape[:3] == DIMS_ORIGINAL_SPACE

        target_affine = affines.rescale_affine(
            mask.affine, shape=DIMS_ORIGINAL_SPACE, zooms=VOXEL_SIZE_MNI_305_2MM, new_shape=DIMS_MNI_305_2MM
        )

        resampled_mask = resample_img(
            mask, target_shape=DIMS_MNI_305_2MM, target_affine=target_affine, interpolation='nearest'
        )

        out_path = get_graymatter_mask_path(subject, downsampled=True)

        nib.save(resampled_mask, out_path)


def downsample_functional_scans(args):
    print("downsampling functional scans")
    for subject in args.subjects:
        print(subject)

        img_paths = glob(
            os.path.join(
                FMRI_PREPROCESSED_DATA_DIR, 'preprocess_workflow', nipype_subject_id(subject), '*', 'coregister',
                '*bold.nii'
            )
        )
        for img_path in tqdm(img_paths):
            img = nib.load(img_path)
            assert img.shape[:3] == DIMS_ORIGINAL_SPACE

            target_affine = affines.rescale_affine(
                img.affine, shape=DIMS_ORIGINAL_SPACE, zooms=VOXEL_SIZE_MNI_305_2MM, new_shape=DIMS_MNI_305_2MM
            )

            resampled_img = resample_img(img, target_shape=DIMS_MNI_305_2MM, target_affine=target_affine)

            out_path = img_path.replace("coregister", "coregister_downsampled")
            assert out_path != img_path

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            nib.save(resampled_img, out_path)

            # from nilearn import plotting
            # template = load_mni152_template(resolution=2)
            # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
            # plotting.plot_stat_map(
            #     img,
            #     bg_img=template,
            #     cut_coords=(36, -27, 66),
            #     threshold=100,
            #     title="img in original resolution",
            #     axes=axes[0],
            # )
            # plotting.plot_stat_map(
            #     resampled_img,
            #     bg_img=template,
            #     cut_coords=(36, -27, 66),
            #     threshold=100,
            #     title="Resampled img",
            #     axes=axes[1],
            # )
            # plt.show()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    downsample_graymatter_masks(args)
    downsample_functional_scans(args)
