import argparse
import warnings

import numpy as np
from matplotlib.patches import Circle
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle

from nilearn.image import resample_to_img
from nilearn.surface import surface
from scipy import stats
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS

from utils import VISION_MEAN_FEAT_KEY, RESULTS_DIR, SUBJECTS

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")

COLORBAR_MAX = 0.85
COLORBAR_THRESHOLD_MIN = 0.6
COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.03
VIEWS = ["lateral", "medial"]  # , "ventral"]   #, "ventral"]

HEMIS = ['left', 'right']

BASE_METRICS = ["test_overall", "test_captions", "test_images"]
CHANCE_VALUES = {"captions": 0.5,
                 "images": 0.5,
                 "mean(imgs,captions)": 0.5,
                 "min(imgs,captions)": 0.5,
                 'mean(imgs_agno,captions_agno)-mean(imgs_specific,captions_specific)': 0,
                 'imgs_agno - imgs_specific': 0,
                 'captions_agno - captions_specific': 0,
                 'imgs_agno - imgs_specific (cross)': 0,
                 'captions_agno - captions_specific (cross)': 0,
                 'mean(imgs_agno,captions_agno)-mean(imgs_specific,captions_specific) (cross)': 0,
                 'mean(captions_agno - captions_specific, imgs_agno - imgs_specific)': 0,
                 'min(captions_agno - captions_specific, imgs_agno - imgs_specific)': 0,
                 }


def add_to_all_scores(all_scores, scores):
    for hemi in HEMIS:
        for score_name in scores[hemi].keys():
            if score_name not in all_scores[hemi]:
                all_scores[hemi][score_name] = scores[hemi][score_name].reshape(-1, 1)
            else:
                all_scores[hemi][score_name] = np.concatenate(
                    (scores[hemi][score_name].reshape(-1, 1), all_scores[hemi][score_name]), axis=1)


def correlation_num_voxels_acc(scores_data, scores, hemi, nan_locations):
    sns.histplot(x=scores_data["n_neighbors"], y=scores[hemi]["overall"][~nan_locations])
    plt.xlabel("number of voxels")
    plt.ylabel("pairwise accuracy (mean)")
    corr = pearsonr(scores_data["n_neighbors"], scores[hemi]["overall"][~nan_locations])
    plt.title(f"pearson r: {corr[0]:.2f} | p = {corr[1]}")
    plt.savefig("results/searchlight_correlation_num_voxels_acc.png", dpi=300)

    plt.figure()
    sns.regplot(x=scores_data["n_neighbors"], y=scores[hemi]["overall"][~nan_locations], x_bins=30)
    plt.xlabel("number of voxels")
    plt.ylabel("pairwise accuracy (mean)")
    corr = pearsonr(scores_data["n_neighbors"], scores[hemi]["overall"][~nan_locations])
    plt.title(f"pearson r: {corr[0]:.2f} | p = {corr[1]}")
    plt.savefig("results/searchlight_correlation_num_voxels_acc_binned.png", dpi=300)


def run(args):
    per_subject_scores = []
    all_subjects = set()
    all_scores = {hemi: dict() for hemi in HEMIS}
    t_values = {hemi: dict() for hemi in HEMIS}

    resolution = "fsaverage6"
    alpha = 1

    results_regex = os.path.join(SEARCHLIGHT_OUT_DIR,
                                 f'train/{args.model}/*/*/{resolution}/left/{args.mode}/alpha_{str(alpha)}.p')
    results_paths = np.array(sorted(glob(results_regex)))

    for path in results_paths:
        subject = os.path.dirname(path).split("/")[-4]
        alpha = float(os.path.basename(path).split("_")[1][:-2])

        scores = dict()
        print(path)
        for hemi in HEMIS:

            scores[hemi] = dict()
            path_scores_hemi = path.replace('left', hemi)
            if os.path.isfile(path_scores_hemi):
                scores_data = pickle.load(open(path_scores_hemi, 'rb'))
                nan_locations = scores_data['nan_locations']
                scores_hemi = scores_data['scores']

                for metric in BASE_METRICS:
                    score_name = metric.split("_")[1]
                    scores[hemi][score_name] = np.repeat(np.nan, nan_locations.shape)
                    scores[hemi][score_name][~nan_locations] = np.array([score[metric] for score in scores_hemi])

                # correlation_num_voxels_acc(scores_data, scores, hemi, nan_locations)
                print(hemi, {n: round(np.nanmean(score), 4) for n, score in scores[hemi].items()})
                print(hemi, {f"{n}_max": round(np.nanmax(score), 2) for n, score in scores[hemi].items()})
                scores[hemi]["mean(imgs,captions)"] = scores[hemi]["overall"]
                del scores[hemi]["overall"]
                scores[hemi]["min(imgs,captions)"] = np.min(
                    (scores[hemi]['images'], scores[hemi]['captions']), axis=0)

                path_scores_hemi_captions = path_scores_hemi.replace('train/', 'train_captions/')
                scores_mod_specific_captions = dict()
                if os.path.isfile(path_scores_hemi_captions):
                    scores_hemi_captions = pickle.load(open(path_scores_hemi_captions, 'rb'))['scores']
                    for metric in BASE_METRICS:
                        score_name = metric.split("_")[1]
                        scores_mod_specific_captions[score_name] = np.repeat(np.nan, nan_locations.shape)
                        scores_mod_specific_captions[score_name][~nan_locations] = np.array(
                            [score[metric] for score in scores_hemi_captions])

                path_scores_hemi_images = path_scores_hemi.replace('train/', 'train_images/')
                scores_mod_specific_images = dict()
                if os.path.isfile(path_scores_hemi_images):
                    scores_hemi_images = pickle.load(open(path_scores_hemi_images, 'rb'))['scores']
                    for metric in BASE_METRICS:
                        score_name = metric.split("_")[1]
                        scores_mod_specific_images[score_name] = np.repeat(np.nan, nan_locations.shape)
                        scores_mod_specific_images[score_name][~nan_locations] = np.array(
                            [score[metric] for score in scores_hemi_images])

                if len(scores_mod_specific_captions) > 0 and len(scores_mod_specific_images) > 0:
                    scores[hemi]['imgs_agno - imgs_specific'] = np.array([ai - si for ai, ac, si, sc in
                                                                          zip(scores[hemi]['images'],
                                                                              scores[hemi]['captions'],
                                                                              scores_mod_specific_images['images'],
                                                                              scores_mod_specific_captions[
                                                                                  'captions'])])
                    scores[hemi]['captions_agno - captions_specific'] = np.array([ac - sc for ai, ac, si, sc in
                                                                                  zip(scores[hemi]['images'],
                                                                                      scores[hemi]['captions'],
                                                                                      scores_mod_specific_images[
                                                                                          'images'],
                                                                                      scores_mod_specific_captions[
                                                                                          'captions'])])

                    # scores[hemi]['imgs_agno - imgs_specific (cross)'] = np.array([ai - si for ai, ac, si, sc in
                    #                                                               zip(scores[hemi]['images'],
                    #                                                                   scores[hemi]['captions'],
                    #                                                                   scores_mod_specific_captions[
                    #                                                                       'images'],
                    #                                                                   scores_mod_specific_images[
                    #                                                                       'captions'])])
                    # scores[hemi]['captions_agno - captions_specific (cross)'] = np.array([ac - sc for ai, ac, si, sc in
                    #                                                                       zip(scores[hemi][
                    #                                                                               'images'],
                    #                                                                           scores[hemi][
                    #                                                                               'captions'],
                    #                                                                           scores_mod_specific_captions[
                    #                                                                               'images'],
                    #                                                                           scores_mod_specific_images[
                    #                                                                               'captions'])])

        add_to_all_scores(all_scores, scores)

        print("")

        all_subjects.add(subject)
        scores["subject"] = subject

        per_subject_scores.append(scores)

    all_scores_all_subjects = np.concatenate([all_scores[hemi]["mean(imgs,captions)"] for hemi in HEMIS], axis=1)
    print(f"\n\nOverall mean: {np.nanmean(all_scores_all_subjects):.2f} (stddev: {np.nanstd(all_scores_all_subjects):.2f})")
    print(f"Overall max: {np.nanmax(all_scores_all_subjects):.2f}")

    # calc averages and t-values
    for hemi in HEMIS:
        for score_name in all_scores[hemi].keys():
            alternative = "greater" if CHANCE_VALUES[score_name] == 0.5 else "two-sided"
            popmean = CHANCE_VALUES[score_name]
            enough_data = [(~np.isnan(x)).sum() == len(SUBJECTS) for x in all_scores[hemi][score_name]]
            t_values[hemi][score_name] = np.array([
                stats.ttest_1samp(x, popmean=popmean, alternative=alternative)[0] if ed else np.nan for x, ed
                in
                zip(all_scores[hemi][score_name], enough_data)])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                all_scores[hemi][score_name] = np.nanmean(all_scores[hemi][score_name], axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_values[hemi]['mean(captions_agno - captions_specific, imgs_agno - imgs_specific)'] = np.nanmean((t_values[hemi]['captions_agno - captions_specific'], t_values[hemi]['imgs_agno - imgs_specific']), axis=0)
            t_values[hemi]['min(captions_agno - captions_specific, imgs_agno - imgs_specific)'] = np.nanmin((t_values[hemi]['captions_agno - captions_specific'], t_values[hemi]['imgs_agno - imgs_specific']), axis=0)

    # plot group-level avg scores
    metrics = ["captions", "images", "mean(imgs,captions)", "min(imgs,captions)",
               'imgs_agno - imgs_specific',
               'captions_agno - captions_specific']

    scores = all_scores
    fig = plt.figure(figsize=(5 * len(VIEWS), len(metrics) * 2))
    subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
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
                        cbar_max = min(np.nanmax(scores_hemi), 99)
                        cbar_min = np.nanmin(scores_hemi)

                    plotting.plot_surf_stat_map(
                        infl_mesh,
                        scores_hemi,
                        hemi=hemi,
                        view=view,
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        axes=axes[i * 2 + j],
                        colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                        threshold=COLORBAR_THRESHOLD_MIN if CHANCE_VALUES[metric] == 0.5 else COLORBAR_DIFFERENCE_THRESHOLD_MIN,
                        vmax=COLORBAR_MAX if CHANCE_VALUES[metric] == 0.5 else None,
                        vmin=0.5 if CHANCE_VALUES[metric] == 0.5 else None,
                        cmap="hot" if CHANCE_VALUES[metric] == 0.5 else "cold_hot",
                        symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
                    )
                    axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.9, fontsize=10)
                else:
                    axes[i * 2 + j].axis('off')

    title = f"{args.model}_{args.mode}_group_level_pairwise_acc"
    # fig.suptitle(title)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')

    # plot group-level t-values
    metrics = ['imgs_agno - imgs_specific',
               'captions_agno - captions_specific',
               'mean(captions_agno - captions_specific, imgs_agno - imgs_specific)',
               'min(captions_agno - captions_specific, imgs_agno - imgs_specific)']
    scores = t_values
    fig = plt.figure(figsize=(5 * len(VIEWS), len(metrics) * 2))
    subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)

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
                        cbar_max = min(np.nanmax(scores_hemi), 99)
                        cbar_min = np.nanmin(scores_hemi)

                    plotting.plot_surf_stat_map(
                        infl_mesh,
                        scores_hemi,
                        hemi=hemi,
                        view=view,
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        axes=axes[i * 2 + j],
                        colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                        threshold=2.571, # for 5 degrees of freedom (6 subjects): 2.571 for p<0.05 (two-sided) | 3.365 for p<0.01(one-sided!)
                        vmax=99 if CHANCE_VALUES[metric] == 0.5 else None,
                        vmin=0.0 if CHANCE_VALUES[metric] == 0.5 else None,
                        cmap="hot" if CHANCE_VALUES[metric] == 0.5 else "cold_hot",
                        symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
                    )
                    axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.9, fontsize=10)
                else:
                    axes[i * 2 + j].axis('off')

    title = f"{args.model}_{args.mode}_group_level_t_values"
    # fig.suptitle(title)
    # fig.tight_layout()
    fig.subplots_adjust(right=0.85, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')

    # per-subject plots
    metrics = ["captions", "images", "mean(imgs,captions)", "min(imgs,captions)",
               'imgs_agno - imgs_specific',
               'captions_agno - captions_specific']
    print("\n\nCreating per-subject plots..")
    for scores in tqdm(per_subject_scores):
        fig = plt.figure(figsize=(5 * len(VIEWS), len(metrics) * 2))
        subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
        fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)

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
                            axes=axes[i * 2 + j],
                            colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                            threshold=COLORBAR_THRESHOLD_MIN if cbar_min >= 0 else COLORBAR_DIFFERENCE_THRESHOLD_MIN,
                            vmax=COLORBAR_MAX if cbar_min >= 0 else None,  # cbar_max,
                            vmin=0.5 if cbar_min >= 0 else None,
                            cmap="hot" if cbar_min >= 0 else "cold_hot",
                            symmetric_cbar=True if cbar_min < 0 else "auto",
                        )
                        axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.9, fontsize=10)

                        # plotting.plot_surf_contours(infl_mesh, parcellation_surf, labels=labels,
                        #                             levels=regions_indices, axes=row_axes[i*2+j],
                        #                             legend=True,
                        #                             colors=colors)
                    else:
                        axes[i * 2 + j].axis('off')

        title = f"{args.model}_{args.mode}_{scores['subject']}"
        # fig.suptitle(title)
        # fig.tight_layout()
        fig.subplots_adjust(right=0.85, wspace=-0.1, hspace=0, top=1)
        title += f"_alpha_{str(alpha)}"
        results_searchlight = os.path.join(RESULTS_DIR, "searchlight", resolution, f"{title}.png")
        os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
        plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='vilt')
    parser.add_argument("--mode", type=str, default='n_neighbors_100')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
