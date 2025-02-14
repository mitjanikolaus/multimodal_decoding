import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, get_hparam_suffix, \
    add_searchlight_permutation_args
from utils import METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, HEMIS, RESULTS_DIR, METRIC_CROSS_DECODING


def run(args):
    args.metric = METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC
    tfce_values_diff_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
    tfce_values_diff = pickle.load(open(tfce_values_diff_path, 'rb'))
    tfce_values_diff = np.concatenate([tfce_values_diff[hemi][args.metric] for hemi in HEMIS])

    args.metric = METRIC_CROSS_DECODING
    tfce_values_cross_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
    tfce_values_cross = pickle.load(open(tfce_values_cross_path, 'rb'))
    tfce_values_cross = np.concatenate([tfce_values_cross[hemi][args.metric] for hemi in HEMIS])

    plt.scatter(tfce_values_cross, tfce_values_diff)
    plt.xlabel('tfce value for mod agnostic cross')
    plt.ylabel('tfce value for mod agnostic diff')
    corr = pearsonr(tfce_values_cross[~np.isnan(tfce_values_cross)], tfce_values_diff[~np.isnan(tfce_values_cross)])
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_mod_agnostic_regions_methods.png'))


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_searchlight_permutation_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
