import argparse
import os

import nibabel
import numpy as np

from analyses.cluster_analysis import calc_significance_cutoff, create_results_cluster_masks
from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, get_hparam_suffix, \
    add_searchlight_permutation_args
from utils import HEMIS, FS_HEMI_NAMES


def split_clusters(args):
    results_dir = permutation_results_dir(args)

    significance_cutoff, _ = calc_significance_cutoff(args, args.p_value_threshold)

    tfce_vals = dict()
    for hemi in HEMIS:
        tfce_vals_path = os.path.join(results_dir, "results_maps",
                                      f"tfce_values{get_hparam_suffix(args)}_{FS_HEMI_NAMES[hemi]}.gii")
        tfce_vals_hemi = nibabel.load(tfce_vals_path)

        if hemi == 'left':
            # split inferior parietal and middle temporal cluster:
            tfce_vals_hemi.darrays[0].data[51847] = 0

            # split middle temporal and inferior temporal cluster:
            tfce_vals_hemi.darrays[0].data[90608] = 0
            tfce_vals_hemi.darrays[0].data[10035] = 0
            tfce_vals_hemi.darrays[0].data[162057] = 0

        tfce_vals_hemi.darrays[0].data[tfce_vals_hemi.darrays[0].data < significance_cutoff] = 0
        tfce_vals_hemi.darrays[0].data[np.isnan(tfce_vals_hemi.darrays[0].data)] = 0
        tfce_vals[hemi] = tfce_vals_hemi.darrays[0].data

    create_results_cluster_masks(tfce_vals, permutation_results_dir(args), get_hparam_suffix(args), args.metric,
                                 args.resolution, args.radius, args.n_neighbors)


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    split_clusters(args)
