import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, get_hparam_suffix, \
    load_per_subject_scores, add_searchlight_permutation_args
from eval import ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, ACC_IMAGES_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES
from utils import HEMIS, RESULTS_DIR


def run(args):
    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
    tfce_values = pickle.load(open(tfce_values_path, 'rb'))
    subject_scores, nan_locations, n_neighbors = load_per_subject_scores(
        args,
        return_nan_locations_and_n_neighbors=True,
    )

    tfce = np.concatenate([tfce_values[hemi][args.metric] for hemi in HEMIS])
    imagery = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC] for sub in args.subjects], axis=0)
         for hemi in HEMIS]
    )
    tfce_filtered = tfce[~np.isnan(imagery) & (tfce > 0)]
    imagery_filtered = imagery[~np.isnan(imagery) & (tfce > 0)]
    plt.scatter(tfce_filtered, imagery_filtered)
    plt.xlabel('tfce value for mod agnostic regions')
    plt.ylabel('mean imagery decoding accuracy')
    corr = pearsonr(tfce_filtered, imagery_filtered)
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_mod_agnostic_regions.png'))



    cross_images = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS] for sub in args.subjects], axis=0)
         for hemi in HEMIS]
    )
    cross_captions = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_CAPTIONS_MOD_SPECIFIC_IMAGES] for sub in args.subjects], axis=0)
         for hemi in HEMIS]
    )
    acc_cross = np.min((cross_images, cross_captions), axis=0)
    acc_cross_filtered = acc_cross[~np.isnan(imagery)]
    imagery_filtered = imagery[~np.isnan(imagery)]

    plt.scatter(acc_cross_filtered, imagery_filtered)
    plt.xlabel('mean of min cross decoding accuracy')
    plt.ylabel('mean imagery decoding accuracy')
    corr = pearsonr(tfce_filtered, imagery_filtered)
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_cross_decoding.png'))


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_searchlight_permutation_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
