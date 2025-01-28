import argparse
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, get_hparam_suffix, \
    load_per_subject_scores
from data import VISION_FEATS_ONLY, LANG_FEATS_ONLY, SELECT_DEFAULT
from eval import ACC_IMAGERY_WHOLE_TEST
from utils import DEFAULT_RESOLUTION, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, DEFAULT_MODEL, HEMIS, SUBJECTS, RESULTS_DIR


def run(args):
    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
    tfce_values = pickle.load(open(tfce_values_path, 'rb'))
    subject_scores, nan_locations, n_neighbors = load_per_subject_scores(
        args,
        return_nan_locations_and_n_neighbors=True,
    )

    tfce = np.concatenate([tfce_values[hemi][args.metric] for hemi in HEMIS])
    imagery = np.concatenate(
        [np.mean([subject_scores[sub][hemi][ACC_IMAGERY_WHOLE_TEST] for sub in args.subjects], axis=0) for hemi in
         HEMIS])
    plt.scatter(tfce, imagery)
    plt.xlabel('tfce value for mod agnostic advantage')
    plt.ylabel('mean imagery decoding accuracy')
    corr = pearsonr(tfce[~np.isnan(imagery)], imagery[~np.isnan(imagery)])
    plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_mod_agnostic_regions.png'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT)

    parser.add_argument("--mod-specific-images-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-images-features", type=str, default=VISION_FEATS_ONLY)
    parser.add_argument("--mod-specific-images-test-features", type=str, default=VISION_FEATS_ONLY)

    parser.add_argument("--mod-specific-captions-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-captions-features", type=str, default=LANG_FEATS_ONLY)
    parser.add_argument("--mod-specific-captions-test-features", type=str, default=LANG_FEATS_ONLY)

    parser.add_argument("--mode", type=str, default='n_neighbors_750')
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)
    parser.add_argument("--tfce-dh", type=float, default=0.1)
    parser.add_argument("--tfce-clip", type=float, default=100)

    parser.add_argument("--metric", type=str, default=METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
