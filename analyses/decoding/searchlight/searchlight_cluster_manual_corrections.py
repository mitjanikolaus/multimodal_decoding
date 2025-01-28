import argparse
import os

import nibabel
import numpy as np

from analyses.cluster_analysis import calc_significance_cutoff, compute_results_clusters
from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, get_hparam_suffix
from utils import HEMIS, FS_HEMI_NAMES, DEFAULT_RESOLUTION, SUBJECTS, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, \
    DEFAULT_MODEL


def split_clusters(args):
    results_dir = permutation_results_dir(args)

    significance_cutoff, _ = calc_significance_cutoff(args, args.p_value_threshold)

    tfce_vals = dict()
    for hemi in HEMIS:
        tfce_vals_path = os.path.join(results_dir, "results_maps", f"tfce_values{get_hparam_suffix(args)}_{FS_HEMI_NAMES[hemi]}.gii")
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

    compute_results_clusters(tfce_vals, args)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--features", type=str, default="avg_test_avg")

    parser.add_argument("--mod-specific-vision-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-vision-features", type=str, default="vision_test_vision")

    parser.add_argument("--mod-specific-lang-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-lang-features", type=str, default="lang_test_lang")

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--mode", type=str, default='n_neighbors_750')
    parser.add_argument("--per-subject-plots", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot-null-distr", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--metric", type=str, default=METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    split_clusters(args)
