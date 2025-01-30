import argparse
import numpy as np
import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from analyses.decoding.searchlight.searchlight_permutation_testing import load_per_subject_scores, \
    permutation_results_dir, add_searchlight_permutation_args
from eval import ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY_WHOLE_TEST, ACC_CROSS_IMAGES_TO_CAPTIONS, \
    ACC_CROSS_CAPTIONS_TO_IMAGES, ACC_IMAGERY
from utils import HEMIS, export_to_gifti, FS_HEMI_NAMES, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES, \
    METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_DECODING


def plot_correlation_num_voxels_acc(scores, nan_locations, n_neighbors, results_dir, args):
    all_scores = []
    all_neighbors = []
    for subject in args.subjects:
        for hemi in HEMIS:
            for metric in [ACC_CAPTIONS, ACC_IMAGES]:
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
    subject_scores, nan_locations, n_neighbors = load_per_subject_scores(
        args,
        return_nan_locations_and_n_neighbors=True,
    )
    if n_neighbors[args.subjects[0]][HEMIS[0]] is not None:
        create_n_vertices_gifti(nan_locations, n_neighbors, results_dir, args)
        plot_correlation_num_voxels_acc(subject_scores, nan_locations, n_neighbors, results_dir, args)

    METRICS = [ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES,
               ACC_CROSS_IMAGES_TO_CAPTIONS, ACC_CROSS_CAPTIONS_TO_IMAGES]

    subject_scores_avgd = dict()
    for metric in METRICS:
        for hemi in HEMIS:
            subject_scores_avgd[hemi] = dict()
            for subj in args.subjects:
                if metric in subject_scores[subj][hemi]:
                    score_hemi = subject_scores[subj][hemi][metric]
                    path_out = os.path.join(results_dir, subj, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                    os.makedirs(os.path.dirname(path_out), exist_ok=True)
                    export_to_gifti(score_hemi, path_out)
                else:
                    print(f"missing metric: {subj} {metric} {hemi}")

            if metric in subject_scores[args.subjects[-1]][hemi]:
                subject_scores_avgd[hemi][metric] = np.mean(  # TODO at least 3 datapoints?
                    [subject_scores[subj][hemi][metric] for subj in args.subjects], axis=0)
                print(f"{metric} ({hemi} hemi) mean over subjects: {np.mean(subject_scores_avgd[hemi][metric])}")
                path_out = os.path.join(results_dir, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                export_to_gifti(subject_scores_avgd[hemi][metric], path_out)
            else:
                print(f"missing metric: {subj} {metric} {hemi}")

    for hemi in HEMIS:
        for subj in args.subjects:
            subject_scores[subj][hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC] = np.nanmin(
                (
                    subject_scores[subj][hemi][METRIC_DIFF_CAPTIONS],
                    subject_scores[subj][hemi][METRIC_DIFF_IMAGES],
                    subject_scores[subj][hemi][ACC_IMAGES],
                    subject_scores[subj][hemi][ACC_CAPTIONS]),
                axis=0)
            path_out = os.path.join(results_dir, subj,
                                    f"{METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(subject_scores[subj][hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC], path_out)

            subject_scores[subj][hemi][METRIC_CROSS_DECODING] = np.nanmin(
                (subject_scores[subj][hemi][ACC_CROSS_IMAGES_TO_CAPTIONS],
                 subject_scores[subj][hemi][ACC_CROSS_CAPTIONS_TO_IMAGES]),
                axis=0
            )
            path_out = os.path.join(results_dir, subj, f"{METRIC_CROSS_DECODING}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(subject_scores[subj][hemi][METRIC_CROSS_DECODING], path_out)

        subject_scores_avgd[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC] = np.nanmin(
            (
                subject_scores_avgd[hemi][METRIC_DIFF_CAPTIONS],
                subject_scores_avgd[hemi][METRIC_DIFF_IMAGES],
                subject_scores_avgd[hemi][ACC_IMAGES],
                subject_scores_avgd[hemi][ACC_CAPTIONS]),
            axis=0)
        path_out = os.path.join(results_dir, f"{METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(subject_scores_avgd[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC], path_out)

        subject_scores_avgd[hemi][METRIC_CROSS_DECODING] = np.nanmin(
            (subject_scores_avgd[hemi][ACC_CROSS_IMAGES_TO_CAPTIONS],
             subject_scores_avgd[hemi][ACC_CROSS_CAPTIONS_TO_IMAGES]),
            axis=0
        )
        path_out = os.path.join(results_dir, f"{METRIC_CROSS_DECODING}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(subject_scores_avgd[hemi][METRIC_CROSS_DECODING], path_out)


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    create_gifti_results_maps(args)
