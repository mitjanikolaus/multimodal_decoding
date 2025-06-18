import argparse
import numpy as np
import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from analyses.decoding.searchlight.searchlight_permutation_testing import load_per_subject_scores, \
    permutation_results_dir, add_searchlight_permutation_args
from data import TRAINING_MODES, MODALITY_AGNOSTIC, SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS, MODALITY_SPECIFIC_IMAGES, \
    MODALITY_SPECIFIC_CAPTIONS
from eval import ACC_IMAGES_MOD_AGNOSTIC, ACC_CAPTIONS_MOD_AGNOSTIC, ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, \
    ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGERY_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_IMAGES, ACC_IMAGERY_NO_STD_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_IMAGES, ACC_IMAGERY_MOD_SPECIFIC_CAPTIONS, \
    ACC_IMAGERY_NO_STD_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_CAPTIONS, \
    ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_MOD_AGNOSTIC, \
    ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC
from utils import HEMIS, export_to_gifti, FS_HEMI_NAMES, METRIC_CROSS_DECODING, METRIC_MOD_AGNOSTIC_AND_CROSS


# METRICS = [
#     ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS,
#     ACC_IMAGES_MOD_AGNOSTIC, ACC_CAPTIONS_MOD_AGNOSTIC, ACC_IMAGERY_MOD_AGNOSTIC,
#     ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES,
#     ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_MOD_SPECIFIC_IMAGES,
#     ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_IMAGES, ACC_IMAGERY_NO_STD_MOD_SPECIFIC_IMAGES,
#     ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_IMAGES, ACC_IMAGERY_MOD_SPECIFIC_CAPTIONS,
#     ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_NO_STD_MOD_SPECIFIC_CAPTIONS,
#     ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_CAPTIONS
# ]


def plot_correlation_num_voxels_acc(scores, nan_locations, n_neighbors, results_dir, args):
    all_scores = []
    all_neighbors = []
    for subject in args.subjects:
        for hemi in HEMIS:
            for metric in [ACC_CAPTIONS_MOD_AGNOSTIC, ACC_IMAGES_MOD_AGNOSTIC]:
                nans = nan_locations[subject][hemi]
                all_scores.extend(scores[subject][hemi][metric][~nans])
                all_neighbors.extend(n_neighbors[subject][hemi])

    corr = pearsonr(all_neighbors, all_scores)

    df = pd.DataFrame({'n_neighbors': all_neighbors, 'scores': all_scores})
    df['n_neighbors_binned'] = pd.cut(df['n_neighbors'], bins=range(125, 1750, 250), labels=list(range(250, 1550, 250)))

    plt.figure()
    sns.barplot(data=df, x="n_neighbors_binned", y="scores")
    plt.xlabel("number of voxels")
    plt.ylabel("pairwise accuracy (mean)")
    plt.savefig(os.path.join(results_dir, "searchlight_correlation_num_voxels_acc.png"),
                dpi=300)
    print(df.groupby('n_neighbors_binned').aggregate({"scores": "mean"}))

    sns.histplot(x=all_neighbors, y=all_scores)
    plt.xlabel("number of voxels")
    plt.ylabel("pairwise accuracy (mean)")
    plt.title(f"pearson r: {corr[0]:.2f} | p = {corr[1]}")
    plt.savefig(os.path.join(results_dir, "searchlight_correlation_num_voxels_acc_hist.png"),
                dpi=300)


def create_n_vertices_gifti(nan_locations, n_neighbors, results_dir, args):
    for hemi in HEMIS:
        all_n_neighbors = []
        for subject in args.subjects:
            neighbors = np.zeros(shape=nan_locations[subject][hemi].shape)
            neighbors[~nan_locations[subject][hemi]] = n_neighbors[subject][hemi]
            all_n_neighbors.append(neighbors)
        all_n_neighbors = np.stack(all_n_neighbors)
        n_neighbors_hemi_avgd = np.nanmean(all_n_neighbors, axis=0)
        path_out = os.path.join(results_dir, f"n_vertices_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(n_neighbors_hemi_avgd, path_out)


def create_gifti_results_maps(args):
    results_dir = os.path.join(permutation_results_dir(args), "acc_results_maps")
    os.makedirs(results_dir, exist_ok=True)

    print("Creating gifti results maps")
    scores = load_per_subject_scores(args)
    # if n_neighbors[args.subjects[0]][HEMIS[0]] is not None:
    #     create_n_vertices_gifti(nan_locations, n_neighbors, results_dir, args)
    #     plot_correlation_num_voxels_acc(subject_scores, nan_locations, n_neighbors, results_dir, args)

    metrics = scores.metric.unique()
    print("Metrics: ", metrics)
    for metric in metrics:
        for hemi in HEMIS:
            for training_mode in TRAINING_MODES:
                for subj in args.subjects:
                    score_hemi_metric = scores[
                        (scores.subject == subj) & (scores.hemi == hemi) & (scores.metric == metric) & (
                                scores.training_mode == training_mode)
                        ]
                    path_out = os.path.join(results_dir, subj, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                    os.makedirs(os.path.dirname(path_out), exist_ok=True)
                    print(f'saving {path_out} ({len(score_hemi_metric)} vertices)')
                    export_to_gifti(score_hemi_metric.value.values, path_out)

                score_hemi_metric_avgd = scores[
                    (scores.hemi == hemi) & (scores.metric == metric) & (scores.training_mode == training_mode)
                    ]
                score_hemi_metric_avgd = score_hemi_metric_avgd.groupby('vertex').aggregate({'value': 'mean'}).value.values
                print(f"{metric} ({hemi} hemi) mean over subjects: {np.nanmean(score_hemi_metric_avgd)}")
                path_out = os.path.join(results_dir, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                print(score_hemi_metric_avgd)
                print(f'saving {path_out} ({len(score_hemi_metric_avgd)} vertices)')
                export_to_gifti(score_hemi_metric_avgd, path_out)

    for hemi in HEMIS:
        for subj in args.subjects:
            sc = scores[(scores.subject == subj) & (scores.hemi == hemi)]

            mod_agnostic_and_cross = np.nanmin(
                (sc[(sc.training_mode == MODALITY_AGNOSTIC) & (sc.metric == SPLIT_TEST_IMAGES)].value.values,
                 sc[(sc.training_mode == MODALITY_SPECIFIC_IMAGES) & (sc.metric == SPLIT_TEST_CAPTIONS)].value.values,
                 sc[(sc.training_mode == MODALITY_AGNOSTIC) & (sc.metric == SPLIT_TEST_CAPTIONS)].value.values,
                 sc[(sc.training_mode == MODALITY_SPECIFIC_CAPTIONS) & (sc.metric == SPLIT_TEST_IMAGES)].value.values),
                axis=0
            )
            path_out = os.path.join(results_dir, subj, f"{METRIC_MOD_AGNOSTIC_AND_CROSS}_{FS_HEMI_NAMES[hemi]}.gii")
            print(f'saving {path_out} ({len(mod_agnostic_and_cross)} vertices)')
            export_to_gifti(mod_agnostic_and_cross, path_out)


        # subject_scores_avgd[hemi][METRIC_MOD_AGNOSTIC_AND_CROSS] = np.nanmin(
        #     (
        #         subject_scores_avgd[hemi][ACC_IMAGES_MOD_AGNOSTIC],
        #         subject_scores_avgd[hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS],
        #         subject_scores_avgd[hemi][ACC_CAPTIONS_MOD_AGNOSTIC],
        #         subject_scores_avgd[hemi][ACC_CAPTIONS_MOD_SPECIFIC_IMAGES]),
        #     axis=0)
        # path_out = os.path.join(results_dir, f"{METRIC_MOD_AGNOSTIC_AND_CROSS}_{FS_HEMI_NAMES[hemi]}.gii")
        # print(f'saving {path_out} ({len(score_hemi_metric_avgd)} vertices)')
        # export_to_gifti(subject_scores_avgd[hemi][METRIC_MOD_AGNOSTIC_AND_CROSS], path_out)


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    create_gifti_results_maps(args)
