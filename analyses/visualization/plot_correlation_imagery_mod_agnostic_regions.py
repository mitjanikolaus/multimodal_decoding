import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, get_hparam_suffix, \
    load_per_subject_scores, add_searchlight_permutation_args
from eval import ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, ACC_IMAGES_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS
from utils import HEMIS, RESULTS_DIR


def run(args):
    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")

    subject_scores, nan_locations, n_neighbors = load_per_subject_scores(
        args,
        return_nan_locations_and_n_neighbors=True,
    )
    imagery = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC] for sub in args.subjects], axis=0)
         for hemi in HEMIS]
    )
    imagery_filtered = imagery[~np.isnan(imagery)]

    # tfce_values = pickle.load(open(tfce_values_path, 'rb'))
    # tfce = np.concatenate([tfce_values[hemi][args.metric] for hemi in HEMIS])
    # tfce_filtered = tfce[~np.isnan(imagery) & (tfce > 0)]
    # imagery_filtered = imagery[~np.isnan(imagery) & (tfce > 0)]
    # plt.scatter(tfce_filtered, imagery_filtered)
    # plt.xlabel('tfce value for mod agnostic regions')
    # plt.ylabel('mean imagery decoding accuracy')
    # corr = pearsonr(tfce_filtered, imagery_filtered)
    # plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    # plt.tight_layout()
    # plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_mod_agnostic_regions.png'))

    within_images = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_IMAGES_MOD_SPECIFIC_IMAGES] for sub in args.subjects], axis=0)
         for hemi in HEMIS]
    )[~np.isnan(imagery)]
    within_captions = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS] for sub in args.subjects], axis=0)
         for hemi in HEMIS]
    )[~np.isnan(imagery)]

    cross_images = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS] for sub in args.subjects], axis=0)
         for hemi in HEMIS]
    )[~np.isnan(imagery)]
    cross_captions = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_CAPTIONS_MOD_SPECIFIC_IMAGES] for sub in args.subjects], axis=0)
         for hemi in HEMIS]
    )[~np.isnan(imagery)]
    cross_min = np.min((cross_images, cross_captions, within_images, within_captions), axis=0)

    plt.figure()
    plt.scatter(cross_min, imagery_filtered, alpha=0.2)
    plt.xlabel('min cross decoding accuracy')
    plt.ylabel('imagery decoding accuracy')
    corr = pearsonr(cross_min, imagery_filtered)
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_cross_decoding.png'))

    plt.figure()
    plt.scatter(cross_images, imagery_filtered, alpha=0.2)
    plt.xlabel('image cross decoding accuracy')
    plt.ylabel('imagery decoding accuracy')
    corr = pearsonr(cross_images, imagery_filtered)
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_cross_decoding_images.png'))

    plt.figure()
    plt.scatter(cross_captions, imagery_filtered, alpha=0.2)
    plt.xlabel('caption cross decoding accuracy')
    plt.ylabel('imagery decoding accuracy')
    corr = pearsonr(cross_captions, imagery_filtered)
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_cross_decoding_captions.png'))

    plt.figure()
    plt.scatter(within_images, imagery_filtered, alpha=0.2)
    plt.xlabel('image decoding accuracy')
    plt.ylabel('imagery decoding accuracy')
    corr = pearsonr(within_images, imagery_filtered)
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_within_decoding_images.png'))

    plt.figure()
    plt.scatter(within_captions, imagery_filtered, alpha=0.2)
    plt.xlabel('caption decoding accuracy')
    plt.ylabel('imagery decoding accuracy')
    corr = pearsonr(within_captions, imagery_filtered)
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_within_decoding_captions.png'))


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_searchlight_permutation_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
