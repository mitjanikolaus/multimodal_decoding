import argparse
import os
import pickle

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from analyses.decoding.searchlight.searchlight_permutation_testing import load_per_subject_scores, \
    add_searchlight_permutation_args, permutation_results_dir, get_hparam_suffix
from eval import ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, \
    ACC_CAPTIONS_MOD_SPECIFIC_IMAGES, ACC_IMAGES_MOD_AGNOSTIC, ACC_CAPTIONS_MOD_AGNOSTIC, \
    ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS
from utils import HEMIS, RESULTS_DIR, METRIC_MOD_AGNOSTIC_AND_CROSS


def run(args):
    for hemis in [['left'], ['right'], HEMIS]:
        hemis_string = "both" if hemis == HEMIS else hemis[0]
        print(f'\nHEMIS: {hemis_string}')

        tfce_scores = dict()
        filter = None
        for metric in [ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, METRIC_MOD_AGNOSTIC_AND_CROSS]:
            args.metric = metric
            tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
            tfce_values = pickle.load(open(tfce_values_path, 'rb'))
            tfce_values = np.concatenate([tfce_values[hemi][args.metric] for hemi in hemis])

            print(f'{metric} tfce nan locs: {np.mean(np.isnan(tfce_values))}')
            print(f'{metric} 0 locs: {np.mean(tfce_values == 0)}')
            if metric == ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC:
                filter = ~np.isnan(tfce_values) & (tfce_values > 0)

            # tfce_values_filtered = tfce_values[filter]
            tfce_scores[metric] = tfce_values#tfce_values_filtered

        # imagery_filtered = imagery[filter & (tfce > 0)]
        # plt.scatter(tfce_filtered, imagery_filtered)
        # plt.xlabel('tfce value for mod agnostic regions')
        # plt.ylabel('mean imagery decoding accuracy')
        # corr = pearsonr(tfce_filtered, imagery_filtered)
        # plt.title(f'pearson r: {corr[0]:.2f} p={corr[1]:.4f}')
        # plt.tight_layout()
        # plt.savefig(os.path.join(RESULTS_DIR, f'corr_imagery_mod_agnostic_regions.png'))

        # mod_agnostic_images = np.concatenate(
        #     [np.mean([subject_scores[sub][hemi][ACC_IMAGES_MOD_AGNOSTIC] for sub in args.subjects], axis=0)
        #      for hemi in hemis]
        # )
        # mod_agnostic_images = mod_agnostic_images[filter]

        # nan_locs = np.isnan(mod_agnostic_images)
        # filter = ~(np.isnan(imagery) | nan_locs)

        # mod_agnostic_captions = np.concatenate(
        #     [np.mean([subject_scores[sub][hemi][ACC_CAPTIONS_MOD_AGNOSTIC] for sub in args.subjects], axis=0)
        #      for hemi in hemis]
        # )[filter]
        #
        # mod_specific_images = np.concatenate(
        #     [np.mean([subject_scores[sub][hemi][ACC_IMAGES_MOD_SPECIFIC_IMAGES] for sub in args.subjects], axis=0)
        #      for hemi in hemis]
        # )[filter]
        # mod_specific_captions = np.concatenate(
        #     [np.mean([subject_scores[sub][hemi][ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS] for sub in args.subjects], axis=0)
        #      for hemi in hemis]
        # )[filter]
        #
        # cross_images = np.concatenate(
        #     [np.mean([subject_scores[sub][hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS] for sub in args.subjects], axis=0)
        #      for hemi in hemis]
        # )[filter]
        # cross_captions = np.concatenate(
        #     [np.mean([subject_scores[sub][hemi][ACC_CAPTIONS_MOD_SPECIFIC_IMAGES] for sub in args.subjects], axis=0)
        #      for hemi in hemis]
        # )[filter]
        # # cross_min = np.min((cross_images, cross_captions), axis=0)
        # mod_agnostic_regions_metric = np.min((cross_images, cross_captions, mod_agnostic_images, mod_agnostic_captions),
        #                                      axis=0)

        for comparison_metric in [METRIC_MOD_AGNOSTIC_AND_CROSS]:
            scatter_kws = {'alpha': 0.1, 's': 1}
            plt.figure()
            sns.regplot(x=tfce_scores[comparison_metric], y=tfce_scores[ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC], color='black', scatter_kws=scatter_kws)
            plt.xlabel(comparison_metric)
            plt.ylabel('imagery decoding accuracy')
            corr = pearsonr(tfce_scores[comparison_metric], tfce_scores[ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC])
            plt.title(f'pearson r: {corr[0]:.2f}')
            plt.tight_layout()
            name = f'corr_imagery_{comparison_metric}_{hemis_string}.png'
            plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
            print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')

        # plt.figure()
        # sns.regplot(x=cross_images, y=imagery_tfce_filtered, color='black', scatter_kws=scatter_kws)
        # plt.xlabel('image cross decoding accuracy')
        # plt.ylabel('imagery decoding accuracy')
        # corr = pearsonr(cross_images, imagery_tfce_filtered)
        # plt.title(f'pearson r: {corr[0]:.2f}')
        # plt.tight_layout()
        # name = f'corr_imagery_cross_decoding_images_{hemis_string}.png'
        # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
        # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')
        #
        # plt.figure()
        # sns.regplot(x=cross_captions, y=imagery_tfce_filtered, color='black', scatter_kws=scatter_kws)
        # plt.xlabel('caption cross decoding accuracy')
        # plt.ylabel('imagery decoding accuracy')
        # corr = pearsonr(cross_captions, imagery_tfce_filtered)
        # plt.title(f'pearson r: {corr[0]:.2f}')
        # plt.tight_layout()
        # name = f'corr_imagery_cross_decoding_captions_{hemis_string}.png'
        # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
        # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')
        #
        # plt.figure()
        # sns.regplot(x=mod_agnostic_images, y=imagery_tfce_filtered, color='black', scatter_kws=scatter_kws)
        # plt.xlabel('image decoding accuracy')
        # plt.ylabel('imagery decoding accuracy')
        # corr = pearsonr(mod_agnostic_images, imagery_tfce_filtered)
        # plt.title(f'pearson r: {corr[0]:.2f}')
        # plt.tight_layout()
        # name = f'corr_imagery_mod_agnostic_decoder_images_{hemis_string}.png'
        # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
        # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')
        #
        # plt.figure()
        # sns.regplot(x=mod_agnostic_captions, y=imagery_tfce_filtered, color='black', scatter_kws=scatter_kws)
        # plt.xlabel('caption decoding accuracy')
        # plt.ylabel('imagery decoding accuracy')
        # corr = pearsonr(mod_agnostic_captions, imagery_tfce_filtered)
        # plt.title(f'pearson r: {corr[0]:.2f}')
        # plt.tight_layout()
        # name = f'corr_imagery_mod_agnostic_decoder_captions_{hemis_string}.png'
        # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
        # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')
        #
        # plt.figure()
        # sns.regplot(x=mod_specific_images, y=imagery_tfce_filtered, color='black', scatter_kws=scatter_kws)
        # plt.xlabel('mod specific image decoding accuracy')
        # plt.ylabel('imagery decoding accuracy')
        # corr = pearsonr(mod_specific_images, imagery_tfce_filtered)
        # plt.title(f'pearson r: {corr[0]:.2f}')
        # plt.tight_layout()
        # name = f'corr_imagery_mod_specific_decoder_images_{hemis_string}.png'
        # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
        # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')
        #
        # plt.figure()
        # sns.regplot(x=mod_specific_captions, y=imagery_tfce_filtered, color='black', scatter_kws=scatter_kws)
        # plt.xlabel('mod specific caption decoding accuracy')
        # plt.ylabel('imagery decoding accuracy')
        # corr = pearsonr(mod_specific_captions, imagery_tfce_filtered)
        # plt.title(f'pearson r: {corr[0]:.2f}')
        # plt.tight_layout()
        # name = f'corr_imagery_mod_specific_decoder_captions_{hemis_string}.png'
        # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
        # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')

    # mod_agnostic_images = np.concatenate(
    #     [np.mean([subject_scores[sub][hemi][ACC_IMAGES_MOD_AGNOSTIC] for sub in args.subjects], axis=0)
    #      for hemi in HEMIS]
    # )[filter]
    # mod_agnostic_captions = np.concatenate(
    #     [np.mean([subject_scores[sub][hemi][ACC_CAPTIONS_MOD_AGNOSTIC] for sub in args.subjects], axis=0)
    #      for hemi in HEMIS]
    # )[filter]

    # diff_images = mod_agnostic_images - mod_agnostic_images
    # diff_captions = mod_agnostic_captions - mod_agnostic_captions
    #
    # diff_metric = np.min((diff_images, diff_captions), axis=0)
    # plt.figure()
    # sns.regplot(x=diff_metric, y=imagery_filtered, color='black', scatter_kws=scatter_kws)
    # plt.xlabel('performance advantage mod agnostic')
    # plt.ylabel('imagery decoding accuracy')
    # corr = pearsonr(diff_metric, imagery_filtered)
    # plt.title(f'pearson r: {corr[0]:.2f}')
    # plt.tight_layout()
    # name = f'corr_imagery_diff_metric.png'
    # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
    # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')

    # plt.figure()
    # sns.regplot(x=diff_captions, y=imagery_filtered, color='black', scatter_kws=scatter_kws)
    # plt.xlabel('performance advantage mod agnostic for caption decoding')
    # plt.ylabel('imagery decoding accuracy')
    # corr = pearsonr(diff_captions, imagery_filtered)
    # plt.title(f'pearson r: {corr[0]:.2f}')
    # plt.tight_layout()
    # name = f'corr_imagery_diff_metric_captions.png'
    # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
    # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')
    #
    #
    # plt.figure()
    # sns.regplot(x=diff_images, y=imagery_filtered, color='black', scatter_kws=scatter_kws)
    # plt.xlabel('performance advantage mod agnostic for image decoding')
    # plt.ylabel('imagery decoding accuracy')
    # corr = pearsonr(diff_images, imagery_filtered)
    # plt.title(f'pearson r: {corr[0]:.2f}')
    # plt.tight_layout()
    # name = f'corr_imagery_diff_metric_images.png'
    # plt.savefig(os.path.join(RESULTS_DIR, name), dpi=300)
    # print(f'{name} pearson r: {corr[0]:.2f} p={corr[1]:.10f}')


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_searchlight_permutation_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
