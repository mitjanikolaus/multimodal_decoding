import argparse
import numpy as np
import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from analyses.searchlight.searchlight import METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES, METRIC_MIN, METRIC_CAPTIONS, \
    METRIC_IMAGES, \
    SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR
from analyses.searchlight.searchlight_permutation_testing import load_per_subject_scores, permutation_results_dir
from preprocessing.transform_to_surface import DEFAULT_RESOLUTION
from utils import SUBJECTS, HEMIS, export_to_gifti, FS_HEMI_NAMES


def plot_correlation_num_voxels_acc(scores, nan_locations, n_neighbors, results_dir, args):
    all_scores = []
    all_neighbors = []
    for subject in args.subjects:
        for hemi in HEMIS:
            for metric in ["captions", "images"]:
                nans = nan_locations[subject][hemi]
                all_scores.extend(scores[subject][hemi][metric][~nans])
                all_neighbors.extend(n_neighbors[subject][hemi])

    corr = pearsonr(all_neighbors, all_scores)

    df = pd.DataFrame({'n_neighbors': all_neighbors, 'scores': all_scores})
    df['n_neighbors_binned'] = pd.cut(df['n_neighbors'], bins=range(125, 1750, 250),
                                      labels=list(range(250, 1550, 250)))

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


def create_gifti_results_maps(args):
    results_dir = os.path.join(permutation_results_dir(args), "acc_scores_gifti")
    os.makedirs(results_dir, exist_ok=True)

    print("Creating gifti results maps")
    subject_scores, nan_locations, n_neighbors = load_per_subject_scores(
        args,
        return_nan_locations_and_n_neighbors=True,
    )
    if n_neighbors[args.subjects[0]][HEMIS[0]] is not None:
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

        plot_correlation_num_voxels_acc(subject_scores, nan_locations, n_neighbors, results_dir, args)

    METRICS = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES]

    for metric in METRICS:
        for hemi in HEMIS:
            if metric in subject_scores[args.subjects[0]][hemi]:
                score_hemi_avgd = np.nanmean([subject_scores[subj][hemi][metric] for subj in args.subjects], axis=0)
                print(f"{metric} ({hemi} hemi) mean over subjects: {np.nanmean(score_hemi_avgd)}")
                path_out = os.path.join(results_dir, f"{metric.replace(' ', '')}_{FS_HEMI_NAMES[hemi]}.gii")
                export_to_gifti(score_hemi_avgd, path_out)

                for subj in args.subjects:
                    score_hemi = subject_scores[subj][hemi][metric]
                    path_out = os.path.join(results_dir, subj, f"{metric.replace(' ', '')}_{FS_HEMI_NAMES[hemi]}.gii")
                    os.makedirs(os.path.dirname(path_out), exist_ok=True)
                    export_to_gifti(score_hemi, path_out)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--model", type=str, default='imagebind')
    parser.add_argument("--features", type=str, default="avg_test_avg")

    parser.add_argument("--mod-specific-vision-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-vision-features", type=str, default="vision_test_vision")

    parser.add_argument("--mod-specific-lang-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-lang-features", type=str, default="lang_test_lang")

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--mode", type=str, default='n_neighbors_200')

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--metric", type=str, default=METRIC_MIN)

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, exist_ok=True)
    args = get_args()

    create_gifti_results_maps(args)
