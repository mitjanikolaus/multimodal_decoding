import argparse
import os
import pickle

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from analyses.cluster_analysis import get_edge_lengths_dicts_based_on_edges, calc_tfce_values
from analyses.decoding.searchlight.searchlight_permutation_testing import add_searchlight_permutation_args, \
    permutation_results_dir, get_hparam_suffix
from eval import ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, ACC_IMAGES_MOD_SPECIFIC_IMAGES, \
    ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS
from utils import HEMIS, RESULTS_DIR, METRIC_MOD_AGNOSTIC_AND_CROSS, METRIC_CROSS_DECODING

COMPARISON_METRICS = [METRIC_MOD_AGNOSTIC_AND_CROSS, ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS]


def calc_additional_test_statistics(args):
    for metric in COMPARISON_METRICS:
        t_values_path = os.path.join(permutation_results_dir(args), "t_values.p")
        args.metric = metric
        t_values = pickle.load(open(t_values_path, 'rb'))
        tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
        if not os.path.isfile(tfce_values_path):
            print(f"calculating tfce for {metric} ..")
            edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)
            tfce_values = calc_tfce_values(t_values, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e,
                                           dh=args.tfce_dh, use_tqdm=True)
            pickle.dump(tfce_values, open(tfce_values_path, "wb"))


def run(args):
    calc_additional_test_statistics(args)

    for hemis in [[HEMIS[0]], [HEMIS[1]]]:
        hemis_string = "both" if hemis == HEMIS else hemis[0]
        print(f'\nHEMIS: {hemis_string}')

        tfce_scores = dict()
        # filter = None
        for metric in [ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC] + COMPARISON_METRICS:
            args.metric = metric
            tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
            tfce_values = pickle.load(open(tfce_values_path, 'rb'))
            tfce_values = np.concatenate([tfce_values[hemi][args.metric] for hemi in hemis])

            # print(f'{metric} tfce nan locs: {np.mean(np.isnan(tfce_values))}')
            # print(f'{metric} 0 locs: {np.mean(tfce_values == 0)}')
            # if metric == ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC:
            #     filter = ~np.isnan(tfce_values) & (tfce_values > 0)

            # tfce_values_filtered = tfce_values[filter]
            tfce_scores[metric] = tfce_values  # tfce_values_filtered

        for comparison_metric in COMPARISON_METRICS:
            scatter_kws = {'alpha': 0.1, 's': 1}
            plt.figure()
            sns.regplot(x=tfce_scores[comparison_metric], y=tfce_scores[ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC],
                        color='black', scatter_kws=scatter_kws)
            plt.xlabel(comparison_metric)
            plt.ylabel('imagery decoding accuracy')
            corr = pearsonr(tfce_scores[comparison_metric], tfce_scores[ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC])
            plt.title(f'pearson r: {corr[0]:.2f}')
            plt.tight_layout()
            name = f'corr_imagery_{comparison_metric}_{hemis_string}.png'
            plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
            print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_searchlight_permutation_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
