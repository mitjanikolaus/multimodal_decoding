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
from analyses.searchlight_permutation_testing import METRIC_MIN_DIFF_BOTH_MODALITIES, METRIC_DIFF_IMAGES, \
    METRIC_DIFF_CAPTIONS, METRIC_CAPTIONS, METRIC_IMAGES, load_per_subject_scores, CHANCE_VALUES, METRIC_CODES
from utils import RESULTS_DIR, SUBJECTS, HEMIS

VIEWS = ["lateral", "medial", "ventral", "posterior"]
COLORBAR_MAX = 1
COLORBAR_THRESHOLD_MIN = 0.6
COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.01

DEFAULT_T_VALUE_THRESH = 0.824
DEFAULT_TFCE_VAL_THRESH = 10


def plot_test_statistics(test_statistics, args, results_path, filename_suffix=""):
    print(f"plotting test stats {filename_suffix}")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    fig = plt.figure(figsize=(5 * len(VIEWS), len(test_statistics) * 2))
    subfigs = fig.subfigures(nrows=len(test_statistics), ncols=1)
    cbar_max = {stat: None for stat in test_statistics.keys()}
    for subfig, (stat_name, values) in zip(subfigs, test_statistics.items()):
        subfig.suptitle(f'{args.metric} {stat_name}', x=0, horizontalalignment="left")
        axes = subfig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
        for i, view in enumerate(VIEWS):
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
                    cmap="hot",
                )
                axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
    title = f"{args.model}_{args.mode}_metric_{METRIC_CODES[args.metric]}_test_stats{filename_suffix}"
    # fig.suptitle(title)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    results_searchlight = os.path.join(results_path, f"{title}.png")
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()


def run(args):
    results_path = os.path.join(RESULTS_DIR, "searchlight", args.resolution, args.features)
    os.makedirs(results_path, exist_ok=True)

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
    p_values['left'] = 1 - p_values['left']
    p_values['right'] = 1 - p_values['right']

    fig = plt.figure(figsize=(5 * len(VIEWS), 2))
    fig.suptitle(f'{args.metric}: 1-(p_value)', x=0, horizontalalignment="left")
    axes = fig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
    cbar_max = 1
    cbar_min = 0.9
    for i, view in enumerate(VIEWS):
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
                threshold=1 - 0.05,
                vmax=cbar_max,
                vmin=cbar_min,
                cmap="hot",
                symmetric_cbar=False,
            )
            axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
    title = f"{args.model}_{args.mode}_metric_{METRIC_CODES[args.metric]}_p_values"
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    results_searchlight = os.path.join(results_path, f"{title}.png")
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    per_subject_scores = load_per_subject_scores(args.model, args.features, args.resolution, args.mode,
                                                 args.l2_regularization_alpha)
    metrics = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]
    print(f"plotting group-level avg scores.")
    fig = plt.figure(figsize=(5 * len(VIEWS), len(metrics) * 2))
    subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
    for subfig, metric in zip(subfigs, metrics):
        subfig.suptitle(f'{metric}', x=0, horizontalalignment="left")
        axes = subfig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
        cbar_max = None
        for i, view in enumerate(VIEWS):
            for j, hemi in enumerate(['left', 'right']):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    score_hemi_avgd = np.mean([per_subject_scores[subj][hemi][metric] for subj in SUBJECTS], axis=0)
                infl_mesh = fsaverage[f"infl_{hemi}"]
                if cbar_max is None:
                    cbar_max = min(np.nanmax(score_hemi_avgd), 99)

                plotting.plot_surf_stat_map(
                    infl_mesh,
                    score_hemi_avgd,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    axes=axes[i * 2 + j],
                    colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                    threshold=COLORBAR_THRESHOLD_MIN if CHANCE_VALUES[
                                                            metric] == 0.5 else COLORBAR_DIFFERENCE_THRESHOLD_MIN,
                    vmax=COLORBAR_MAX if CHANCE_VALUES[metric] == 0.5 else None,
                    vmin=0.5 if CHANCE_VALUES[metric] == 0.5 else None,
                    cmap="hot" if CHANCE_VALUES[metric] == 0.5 else "cold_hot",
                    symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
                )
                axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
    title = f"{args.model}_{args.mode}_pairwise_acc"
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    results_searchlight = os.path.join(results_path, f"{title}.png")
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    if args.null_distr_plots:
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
                args.mode, f"t_values_null_distribution_metric_{METRIC_CODES[args.metric]}_smoothed_{args.smoothing_iterations}.p"
            )
            t_values_smooth_null_distribution = pickle.load(open(smooth_t_values_null_distribution_path, 'rb'))
        null_distribution_tfce_values_file = os.path.join(
            SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
            args.resolution,
            args.mode,
            f"tfce_values_null_distribution_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
        )
        null_distribution_test_statistic = pickle.load(open(null_distribution_tfce_values_file, 'rb'))

        for i in range(10):
            test_statistics = {"t-values": null_distribution_t_values[i]}
            if t_values_smooth_null_distribution is not None:
                test_statistics["t-values-smoothed"] = t_values_smooth_null_distribution[i]
            test_statistics["tfce-values"] = null_distribution_test_statistic[i]
            plot_test_statistics(test_statistics, args, results_path, filename_suffix=f"_null_distr_{i}")

    if args.per_subject_plots:
        metrics = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]
        print("\n\nCreating per-subject plots..")
        for subject, scores in tqdm(per_subject_scores.items()):
            fig = plt.figure(figsize=(5 * len(VIEWS), len(metrics) * 2))
            subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
            fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

            for subfig, metric in zip(subfigs, metrics):
                subfig.suptitle(f'{metric}', x=0, horizontalalignment="left")
                axes = subfig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
                cbar_max = None
                cbar_min = None
                for i, view in enumerate(VIEWS):
                    for j, hemi in enumerate(['left', 'right']):
                        if metric in scores[hemi].keys():
                            scores_hemi = scores[hemi][metric]

                            infl_mesh = fsaverage[f"infl_{hemi}"]
                            if cbar_max is None:
                                cbar_max = np.nanmax(scores_hemi)
                                cbar_min = np.nanmin(scores_hemi)
                            # print(f" | max score: {cbar_max:.2f}")

                            # destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
                            # parcellation = destrieux_atlas['map_right']
                            # parcellation = destrieux_atlas[f'maps']
                            # parcellation_surf = surface.vol_to_surf(parcellation, fsaverage[f'pial_{hemi}'], interpolation="nearest", radius=5).astype(int) #TODO

                            # these are the regions we want to outline
                            # regions_dict = {'L G_pariet_inf-Angular': 'left angular gyrus'}
                            # regions_dict = {b'G_postcentral': 'Postcentral gyrus',
                            #                 b'G_precentral': 'Precentral gyrus'}

                            # get indices in atlas for these labels
                            # regions_indices = [
                            #     [i for i, l in destrieux_atlas['labels'] if l == region][0]
                            #     for region in regions_dict
                            # ]
                            # regions_indices = [
                            #     np.where(np.array(destrieux_atlas['labels']) == region)[0][0]
                            #     for region in regions_dict
                            # ]
                            # colors = ['g', 'b']
                            #
                            # labels = list(regions_dict.values())

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
                                cmap="hot" if cbar_min >= 0 else "cold_hot",
                                symmetric_cbar=True if cbar_min < 0 else "auto",
                            )
                            axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)

                            # plotting.plot_surf_contours(infl_mesh, parcellation_surf, labels=labels,
                            #                             levels=regions_indices, axes=row_axes[i*2+j],
                            #                             legend=True,
                            #                             colors=colors)
                        else:
                            axes[i * 2 + j].axis('off')

            title = f"{args.model}_{args.mode}_{subject}"
            # fig.suptitle(title)
            # fig.tight_layout()
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
    parser.add_argument("--null-distr-plots", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--smoothing-iterations", type=int, default=0)

    parser.add_argument("--tfce", action="store_true")
    parser.add_argument("--tfce-h", type=float, default=2)
    parser.add_argument("--tfce-e", type=float, default=1)

    parser.add_argument("--metric", type=str, default=METRIC_MIN_DIFF_BOTH_MODALITIES)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.features = get_default_features(args.model) if args.features == FEATS_SELECT_DEFAULT else args.features

    run(args)
