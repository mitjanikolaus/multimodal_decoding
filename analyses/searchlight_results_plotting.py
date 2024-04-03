import argparse
import warnings

import numpy as np
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle

from scipy import stats
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns

from utils import RESULTS_DIR

METRIC_MIN_DIFF_BOTH_MODALITIES = 'min(captions_agno - captions_specific, imgs_agno - imgs_specific)'

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")

COLORBAR_MAX = 0.85
COLORBAR_THRESHOLD_MIN = 0.6
COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.02
VIEWS = ["lateral", "medial", "ventral"]

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


def add_to_all_scores(all_scores, scores, hemi):
    for score_name in scores.keys():
        if score_name not in all_scores[hemi]:
            all_scores[hemi][score_name] = scores[score_name].reshape(-1, 1)
        else:
            all_scores[hemi][score_name] = np.concatenate(
                (scores[score_name].reshape(-1, 1), all_scores[hemi][score_name]), axis=1)


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


def process_scores(scores_agnostic, scores_captions, scores_images):
    scores = dict()
    nan_locations = scores_agnostic['nan_locations']

    for metric in BASE_METRICS:
        score_name = metric.split("_")[1]
        scores[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores[score_name][~nan_locations] = np.array([score[metric] for score in scores_agnostic['scores']])

    # correlation_num_voxels_acc(scores_agnostic, scores, hemi, nan_locations)
    print({n: round(np.nanmean(score), 4) for n, score in scores.items()})
    print({f"{n}_max": round(np.nanmax(score), 2) for n, score in scores.items()})
    scores["mean(imgs,captions)"] = scores["overall"]
    del scores["overall"]
    scores["min(imgs,captions)"] = np.min((scores['images'], scores['captions']), axis=0)

    scores_mod_specific_captions = dict()
    for metric in BASE_METRICS:
        score_name = metric.split("_")[1]
        scores_mod_specific_captions[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores_mod_specific_captions[score_name][~nan_locations] = np.array(
            [score[metric] for score in scores_captions['scores']])

    scores_mod_specific_images = dict()
    for metric in BASE_METRICS:
        score_name = metric.split("_")[1]
        scores_mod_specific_images[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores_mod_specific_images[score_name][~nan_locations] = np.array(
            [score[metric] for score in scores_images['scores']])

    scores['imgs_agno - imgs_specific'] = np.array([ai - si for ai, ac, si, sc in
                                                    zip(scores['images'],
                                                        scores['captions'],
                                                        scores_mod_specific_images['images'],
                                                        scores_mod_specific_captions[
                                                            'captions'])])
    scores['captions_agno - captions_specific'] = np.array([ac - sc for ai, ac, si, sc in
                                                            zip(scores['images'],
                                                                scores['captions'],
                                                                scores_mod_specific_images[
                                                                    'images'],
                                                                scores_mod_specific_captions[
                                                                    'captions'])])

    # scores['imgs_agno - imgs_specific (cross)'] = np.array([ai - si for ai, ac, si, sc in
    #                                                               zip(scores['images'],
    #                                                                   scores['captions'],
    #                                                                   scores_mod_specific_captions[
    #                                                                       'images'],
    #                                                                   scores_mod_specific_images[
    #                                                                       'captions'])])
    # scores['captions_agno - captions_specific (cross)'] = np.array([ac - sc for ai, ac, si, sc in
    #                                                                       zip(scores[
    #                                                                               'images'],
    #                                                                           scores[
    #                                                                               'captions'],
    #                                                                           scores_mod_specific_captions[
    #                                                                               'images'],
    #                                                                           scores_mod_specific_images[
    #                                                                               'captions'])])

    return scores


def run(args):
    per_subject_scores = dict()
    all_scores = {hemi: dict() for hemi in HEMIS}
    t_values = {hemi: dict() for hemi in HEMIS}

    all_scores_null_distr = {hemi: dict() for hemi in HEMIS}
    alpha = 1

    results_regex = os.path.join(SEARCHLIGHT_OUT_DIR,
                                 f'train/{args.model}/*/*/{args.resolution}/*/{args.mode}/alpha_{str(alpha)}.p')
    paths_mod_agnostic = np.array(sorted(glob(results_regex)))
    paths_mod_specific_captions = np.array(sorted(glob(results_regex.replace('train/', 'train_captions/'))))
    paths_mod_specific_images = np.array(sorted(glob(results_regex.replace('train/', 'train_images/'))))
    assert len(paths_mod_agnostic) == len(paths_mod_specific_images) == len(paths_mod_specific_captions)

    for path_agnostic, path_caps, path_imgs in zip(paths_mod_agnostic, paths_mod_specific_captions,
                                                   paths_mod_specific_images):
        print(path_agnostic)
        hemi = os.path.dirname(path_agnostic).split("/")[-2]
        subject = os.path.dirname(path_agnostic).split("/")[-4]

        scores_agnostic = pickle.load(open(path_agnostic, 'rb'))
        scores_captions = pickle.load(open(path_caps, 'rb'))
        scores_images = pickle.load(open(path_imgs, 'rb'))

        scores = process_scores(scores_agnostic, scores_captions, scores_images)
        add_to_all_scores(all_scores, scores, hemi)

        print("")

        if subject not in per_subject_scores.keys():
            per_subject_scores[subject] = dict()
        per_subject_scores[subject][hemi] = scores

        null_distribution_file_name = f"alpha_{str(alpha)}_null_distribution.p"
        null_distribution_agnostic = pickle.load(
            open(os.path.join(os.path.dirname(path_agnostic), null_distribution_file_name), 'rb'))
        null_distribution_agnostic = np.concatenate(null_distribution_agnostic)  # TODO update

        null_distribution_captions = pickle.load(
            open(os.path.join(os.path.dirname(paths_mod_specific_captions), null_distribution_file_name), 'rb'))
        null_distribution_captions = np.concatenate(null_distribution_captions)  # TODO update

        null_distribution_images = pickle.load(
            open(os.path.join(os.path.dirname(paths_mod_specific_images), null_distribution_file_name), 'rb'))
        null_distribution_images = np.concatenate(null_distribution_images)  # TODO update

        for distr, distr_caps, distr_imgs in zip(null_distribution_agnostic, null_distribution_captions, null_distribution_images):
            scores = process_scores(distr, distr_caps, distr_imgs)
            add_to_all_scores(all_scores_null_distr, scores, hemi)

        # print("len(null_distribution): ", len(null_distribution))
        # print("len(null_distribution)[0]: ", len(null_distribution[0]))
        # null_distr_captions = [n["captions"] for n in null_distribution[0]]
        # print(f"mean caps: {np.mean(null_distr_captions)}")
        # null_distr_imgs = [n["images"] for n in null_distribution[0]]
        # print(f"mean imgs: {np.mean(null_distr_imgs)}")
        # print(f"max imgs: {np.max(null_distr_imgs)}")
        # print(f"min imgs: {np.min(null_distr_imgs)}")
        # #

    all_scores_all_subjects = np.concatenate([all_scores[hemi]["mean(imgs,captions)"] for hemi in HEMIS], axis=1)
    print(
        f"\n\nOverall mean: {np.nanmean(all_scores_all_subjects):.2f} (stddev: {np.nanstd(all_scores_all_subjects):.2f})")
    print(f"Overall max: {np.nanmax(all_scores_all_subjects):.2f}")

    # calc averages and t-values
    num_subjects = len(per_subject_scores)
    print(f"Calculating t-values for {num_subjects} subjects.")
    for hemi in HEMIS:
        for score_name in all_scores[hemi].keys():
            popmean = CHANCE_VALUES[score_name]
            enough_data = [(~np.isnan(x)).sum() == num_subjects for x in all_scores[hemi][score_name]]
            t_values[hemi][score_name] = np.array([
                stats.ttest_1samp(x, popmean=popmean, alternative="greater")[0] if ed else np.nan for x, ed
                in
                zip(all_scores[hemi][score_name], enough_data)])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                all_scores[hemi][score_name] = np.nanmean(all_scores[hemi][score_name], axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
                (t_values[hemi]['captions_agno - captions_specific'], t_values[hemi]['imgs_agno - imgs_specific']),
                axis=0)

    print("plotting (t-values) threshold 0.824")
    metrics = ['imgs_agno - imgs_specific',
               'captions_agno - captions_specific',
               METRIC_MIN_DIFF_BOTH_MODALITIES]
    scores = t_values
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
                        cbar_max = min(np.nanmax(scores_hemi), 99)
                        cbar_min = max(np.nanmin(scores_hemi), -99)

                    plotting.plot_surf_stat_map(
                        infl_mesh,
                        scores_hemi,
                        hemi=hemi,
                        view=view,
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        axes=axes[i * 2 + j],
                        colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                        threshold=0.824 if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else 2.015,  # p < 0.05
                        vmax=cbar_max,
                        vmin=0 if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else -cbar_max,
                        cmap="hot" if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else "cold_hot",
                        symmetric_cbar=False if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else True,
                    )
                    axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
                else:
                    axes[i * 2 + j].axis('off')
    title = f"{args.model}_{args.mode}_group_level_t_values_tresh_0.824"
    # fig.suptitle(title)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    # plot group-level avg scores
    metrics = ["captions", "images",
               'imgs_agno - imgs_specific',
               'captions_agno - captions_specific']

    scores = all_scores
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
                        threshold=COLORBAR_THRESHOLD_MIN if CHANCE_VALUES[
                                                                metric] == 0.5 else COLORBAR_DIFFERENCE_THRESHOLD_MIN,
                        vmax=COLORBAR_MAX if CHANCE_VALUES[metric] == 0.5 else None,
                        vmin=0.5 if CHANCE_VALUES[metric] == 0.5 else None,
                        cmap="hot" if CHANCE_VALUES[metric] == 0.5 else "cold_hot",
                        symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
                    )
                    axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
                else:
                    axes[i * 2 + j].axis('off')

    title = f"{args.model}_{args.mode}_group_level_pairwise_acc"
    # fig.suptitle(title)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    if args.per_subject_plots:
        # per-subject plots
        metrics = ["captions", "images", "min(imgs,captions)",
                   'imgs_agno - imgs_specific',
                   'captions_agno - captions_specific']
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
            title += f"_alpha_{str(alpha)}"
            results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
            os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
            plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
            plt.close()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='vilt')
    parser.add_argument("--resolution", type=str, default='fsaverage7')
    parser.add_argument("--mode", type=str, default='n_neighbors_100')
    parser.add_argument("--per-subject-plots", default=False, action=argparse.BooleanOptionalAction)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
