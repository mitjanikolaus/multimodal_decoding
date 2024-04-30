import argparse
import warnings

import numpy as np
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
import pickle

from tqdm import tqdm

from analyses.ridge_regression_decoding import FEATS_SELECT_DEFAULT, get_default_features, FEATURE_COMBINATION_CHOICES
from analyses.searchlight import SEARCHLIGHT_OUT_DIR
from analyses.searchlight_permutation_testing import METRIC_DIFF_IMAGES, \
    METRIC_DIFF_CAPTIONS, METRIC_CAPTIONS, METRIC_IMAGES, load_per_subject_scores, CHANCE_VALUES, METRIC_CODES, \
    load_null_distr_per_subject_scores, METRIC_MIN_ALT, METRIC_AGNOSTIC, METRIC_MIN_DIFF_BOTH_MODALITIES
from utils import RESULTS_DIR, SUBJECTS, HEMIS

DEFAULT_VIEWS = ["lateral", "medial", "ventral"]
COLORBAR_MAX = 1
COLORBAR_THRESHOLD_MIN = 0.55
COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.001

CMAP = "cold_white_hot"
CMAP_POS_ONLY = "black_red_r"

DEFAULT_T_VALUE_THRESH = 1  # 0.824
DEFAULT_TFCE_VAL_THRESH = 100

PLOT_NULL_DISTR_NUM_SAMPLES = 10


def plot_test_statistics(test_statistics, args, results_path, filename_suffix=""):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    if "t-values" in test_statistics:
        t_values = test_statistics['t-values']
        metrics = list(t_values['left'].keys())
        metrics.remove(METRIC_MIN_DIFF_BOTH_MODALITIES)

        print(f"plotting t values for {len(metrics)} metrics {filename_suffix}")
        fig = plt.figure(figsize=(5 * len(args.views), len(metrics) * 2))
        subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
        cbar_max = {metric: None for metric in metrics}
        for subfig, metric in zip(subfigs, metrics):
            subfig.suptitle(f'{metric}', x=0, y=1.1, horizontalalignment="left")
            axes = subfig.subplots(nrows=1, ncols=2 * len(args.views), subplot_kw={'projection': '3d'})
            for i, view in enumerate(args.views):
                for j, hemi in enumerate(HEMIS):
                    scores_hemi = t_values[hemi][metric]
                    infl_mesh = fsaverage[f"infl_{hemi}"]
                    if cbar_max[metric] is None:
                        cbar_max[metric] = np.nanmax(scores_hemi)
                    threshold = DEFAULT_T_VALUE_THRESH
                    plotting.plot_surf_stat_map(
                        infl_mesh,
                        scores_hemi,
                        hemi=hemi,
                        view=view,
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        bg_on_data=True,
                        axes=axes[i * 2 + j],
                        colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                        threshold=threshold,
                        vmax=cbar_max[metric],
                        vmin=0,
                        cmap=CMAP_POS_ONLY,
                    )
                    axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
        title = f"{args.model}_{args.mode}_metric_{METRIC_CODES[args.metric]}_test_stats{filename_suffix}"
        # fig.suptitle(title)
        # fig.tight_layout()
        fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
        results_searchlight = os.path.join(results_path, f"{title}.png")
        plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
        plt.close()

    # plot remaining test stats
    test_statistics_filtered = test_statistics.copy()
    del test_statistics_filtered['t-values']

    print(f"plotting test stats {filename_suffix}")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    fig = plt.figure(figsize=(5 * len(args.views), len(test_statistics) * 2))
    subfigs = fig.subfigures(nrows=len(test_statistics), ncols=1)
    cbar_max = {stat: None for stat in test_statistics.keys()}
    for subfig, (stat_name, values) in zip(subfigs, test_statistics.items()):
        subfig.suptitle(f'{args.metric} {stat_name}', x=0, y=1.1, horizontalalignment="left")
        axes = subfig.subplots(nrows=1, ncols=2 * len(args.views), subplot_kw={'projection': '3d'})
        for i, view in enumerate(args.views):
            for j, hemi in enumerate(HEMIS):
                scores_hemi = values[hemi][args.metric]
                infl_mesh = fsaverage[f"infl_{hemi}"]
                if cbar_max[stat_name] is None:
                    if (stat_name == "t-values-smoothed") and (cbar_max['t-values'] is not None):
                        cbar_max[stat_name] = cbar_max['t-values']
                    else:
                        cbar_max[stat_name] = np.nanmax(scores_hemi)
                threshold = DEFAULT_T_VALUE_THRESH if stat_name.startswith("t-values") else DEFAULT_TFCE_VAL_THRESH
                plotting.plot_surf_stat_map(
                    infl_mesh,
                    scores_hemi,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    axes=axes[i * 2 + j],
                    colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                    threshold=threshold,
                    vmax=cbar_max[stat_name],
                    vmin=0,
                    cmap=CMAP_POS_ONLY,
                )
                axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
    title = f"{args.model}_{args.mode}_metric_{METRIC_CODES[args.metric]}_test_stats2{filename_suffix}"
    # fig.suptitle(title)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    results_searchlight = os.path.join(results_path, f"{title}.png")
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()


def plot_acc_scores(per_subject_scores, args, results_path, filename_suffix=""):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    metrics = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_AGNOSTIC, METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]

    print(f"plotting acc scores. {filename_suffix}")
    fig = plt.figure(figsize=(5 * len(args.views), len(metrics) * 2))
    subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
    for subfig, metric in zip(subfigs, metrics):
        subfig.suptitle(f'{metric}', x=0, y=1.1, horizontalalignment="left")
        axes = subfig.subplots(nrows=1, ncols=2 * len(args.views), subplot_kw={'projection': '3d'})
        cbar_max = None
        for i, view in enumerate(args.views):
            for j, hemi in enumerate(['left', 'right']):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    score_hemi_avgd = np.nanmean([per_subject_scores[subj][hemi][metric] for subj in SUBJECTS], axis=0)
                infl_mesh = fsaverage[f"infl_{hemi}"]
                if cbar_max is None:
                    cbar_max = min(np.nanmax(score_hemi_avgd), 99)
                threshold = COLORBAR_THRESHOLD_MIN
                if CHANCE_VALUES[metric] == 0:
                    threshold = COLORBAR_DIFFERENCE_THRESHOLD_MIN
                plotting.plot_surf_stat_map(
                    infl_mesh,
                    score_hemi_avgd,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    axes=axes[i * 2 + j],
                    colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                    threshold=threshold,
                    vmax=COLORBAR_MAX if CHANCE_VALUES[metric] == 0.5 else None,
                    vmin=0.5 if CHANCE_VALUES[metric] == 0.5 else None,
                    cmap=CMAP_POS_ONLY if CHANCE_VALUES[metric] == 0.5 else CMAP,
                    symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
                )
                axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
    title = f"{args.model}_{args.mode}_pairwise_acc{filename_suffix}"
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    results_searchlight = os.path.join(results_path, f"{title}.png")
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()


def run(args):
    results_path = os.path.join(RESULTS_DIR, "searchlight", args.resolution, args.features)
    os.makedirs(results_path, exist_ok=True)

    per_subject_scores = load_per_subject_scores(args)
    plot_acc_scores(per_subject_scores, args, results_path)

    t_values_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
                                 "t_values.p")
    test_statistics = {"t-values": pickle.load(open(t_values_path, 'rb'))}
    if args.smoothing_iterations > 0:
        t_values_smooth_path = os.path.join(
            SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution,
            args.mode,
            f"t_values_metric_{METRIC_CODES[args.metric]}_smoothed_{args.smoothing_iterations}.p"
        )
        test_statistics["t-values-smoothed"] = pickle.load(open(t_values_smooth_path, 'rb'))
    tfce_values_path = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
        f"tfce_values_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
    )
    test_statistics["tfce-values"] = pickle.load(open(tfce_values_path, 'rb'))
    plot_test_statistics(test_statistics, args, results_path)

    print(f"plotting (p-values)")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    p_values_path = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
        f"p_values_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
    )
    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    p_values['left'][p_values['left'] == 0] = np.nan
    p_values['right'][p_values['right'] == 0] = np.nan
    p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    fig = plt.figure(figsize=(5 * len(args.views), 2))
    fig.suptitle(f'{args.metric}: -log10(p_value)', x=0, y=1.1, horizontalalignment="left")
    axes = fig.subplots(nrows=1, ncols=2 * len(args.views), subplot_kw={'projection': '3d'})
    cbar_max = np.nanmax(np.concatenate((p_values['left'], p_values['right'])))
    cbar_min = 0
    for i, view in enumerate(args.views):
        for j, hemi in enumerate(HEMIS):
            scores_hemi = p_values[hemi]
            infl_mesh = fsaverage[f"infl_{hemi}"]
            plotting.plot_surf_stat_map(
                infl_mesh,
                scores_hemi,
                hemi=hemi,
                view=view,
                bg_map=fsaverage[f"sulc_{hemi}"],
                bg_on_data=True,
                axes=axes[i * 2 + j],
                colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                threshold=1.3,  # -log10(0.05) ~ 1.3
                vmax=cbar_max,
                vmin=cbar_min,
                cmap=CMAP_POS_ONLY,
                symmetric_cbar=False,
            )
            axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
    title = f"{args.model}_{args.mode}_metric_{METRIC_CODES[args.metric]}_p_values"
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    results_searchlight = os.path.join(results_path, f"{title}.png")
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    if args.plot_null_distr:
        print("plotting acc maps for null distribution examples")
        per_subject_scores_null_distr = load_null_distr_per_subject_scores(args)
        for i in range(PLOT_NULL_DISTR_NUM_SAMPLES):
            plot_acc_scores(per_subject_scores_null_distr[i], args, results_path, filename_suffix=f"_null_distr_{i}")

        print("plotting test stats for null distribution examples")
        t_values_null_distribution_path = os.path.join(
            SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
            args.resolution,
            args.mode, f"t_values_null_distribution.p"
        )
        null_distribution_t_values = pickle.load(open(t_values_null_distribution_path, 'rb'))
        t_values_smooth_null_distribution = None
        if args.smoothing_iterations > 0:
            smooth_t_values_null_distribution_path = os.path.join(
                SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
                args.resolution,
                args.mode,
                f"t_values_null_distribution_metric_{METRIC_CODES[args.metric]}_smoothed_{args.smoothing_iterations}.p"
            )
            t_values_smooth_null_distribution = pickle.load(open(smooth_t_values_null_distribution_path, 'rb'))
        null_distribution_tfce_values_file = os.path.join(
            SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
            args.resolution,
            args.mode,
            f"tfce_values_null_distribution_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
        )
        null_distribution_test_statistic = pickle.load(open(null_distribution_tfce_values_file, 'rb'))

        for i in range(PLOT_NULL_DISTR_NUM_SAMPLES):
            test_statistics = {"t-values": null_distribution_t_values[i]}
            if t_values_smooth_null_distribution is not None:
                test_statistics["t-values-smoothed"] = t_values_smooth_null_distribution[i]
            test_statistics["tfce-values"] = null_distribution_test_statistic[i]
            plot_test_statistics(test_statistics, args, results_path, filename_suffix=f"_null_distr_{i}")

    if args.per_subject_plots:
        metrics = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]
        print("\n\nCreating per-subject plots..")
        for subject, scores in tqdm(per_subject_scores.items()):
            fig = plt.figure(figsize=(5 * len(args.views), len(metrics) * 2))
            subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
            fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

            for subfig, metric in zip(subfigs, metrics):
                subfig.suptitle(f'{metric}', x=0, horizontalalignment="left")
                axes = subfig.subplots(nrows=1, ncols=2 * len(args.views), subplot_kw={'projection': '3d'})
                cbar_max = None
                cbar_min = None
                for i, view in enumerate(args.views):
                    for j, hemi in enumerate(['left', 'right']):
                        scores_hemi = scores[hemi][metric]

                        infl_mesh = fsaverage[f"infl_{hemi}"]
                        if cbar_max is None:
                            cbar_max = np.nanmax(scores_hemi)
                            cbar_min = np.nanmin(scores_hemi)

                        plotting.plot_surf_stat_map(
                            infl_mesh,
                            scores_hemi,
                            hemi=hemi,
                            view=view,
                            bg_map=fsaverage[f"sulc_{hemi}"],
                            bg_on_data=True,
                            axes=axes[i * 2 + j],
                            colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                            threshold=COLORBAR_THRESHOLD_MIN if cbar_min >= 0 else COLORBAR_DIFFERENCE_THRESHOLD_MIN,
                            vmax=COLORBAR_MAX if cbar_min >= 0 else None,  # cbar_max,
                            vmin=0.5 if cbar_min >= 0 else None,
                            cmap=CMAP_POS_ONLY if cbar_min >= 0 else CMAP,
                            symmetric_cbar=True if cbar_min < 0 else "auto",
                        )
                        axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)

            title = f"{args.model}_{args.mode}_{subject}"
            fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
            results_searchlight = os.path.join(results_path, f"{title}.png")
            plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
            plt.close()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='vilt')
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default='fsaverage7')
    parser.add_argument("--mode", type=str, default='n_neighbors_100')
    parser.add_argument("--per-subject-plots", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot-null-distr", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--smoothing-iterations", type=int, default=0)

    parser.add_argument("--tfce", action="store_true")
    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--metric", type=str, default=METRIC_MIN_ALT)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.features = get_default_features(args.model) if args.features == FEATS_SELECT_DEFAULT else args.features

    run(args)
