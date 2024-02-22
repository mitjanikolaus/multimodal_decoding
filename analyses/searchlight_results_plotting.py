import argparse
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
from tqdm import tqdm

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS

from utils import VISION_MEAN_FEAT_KEY, RESULTS_DIR, SUBJECTS

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")

COLORBAR_MAX = 0.85
COLORBAR_THRESHOLD_MIN = 0.6
COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.05
VIEWS = ["lateral", "medial"]  # , "ventral"]   #, "ventral"]

HEMIS = ['left', 'right']

BASE_METRICS = ["test_overall", "test_captions", "test_images"]
CHANCE_VALUES = {"overall": 0.5,
                 "captions": 0.5,
                 "images": 0.5,
                 "mean(imgs,captions)": 0.5,
                 "min(imgs,captions)": 0.5,
                 'mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific)': 0,
                 'imgs_agno - imgs_specific': 0,
                 'captions_agno - captions_specific': 0,
                 'imgs_agno - imgs_specific (cross)': 0,
                 'captions_agno - captions_specific (cross)': 0,
                 'mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific) (cross)': 0
                 }


def add_to_all_scores(all_scores, scores):
    for hemi in HEMIS:
        for score_name in scores[hemi].keys():
            if score_name not in all_scores[hemi]:
                all_scores[hemi][score_name] = scores[hemi][score_name].reshape(-1, 1)
            else:
                all_scores[hemi][score_name] = np.concatenate(
                    (scores[hemi][score_name].reshape(-1, 1), all_scores[hemi][score_name]), axis=1)


def run(args):
    per_subject_scores = []
    all_subjects = set()
    all_scores = {hemi: dict() for hemi in HEMIS}

    model_name = "vilt"
    resolution = "fsaverage6"
    mode = "n_neighbors_100"
    alpha = 1

    results_regex = os.path.join(SEARCHLIGHT_OUT_DIR, f'train/{model_name}/*/*/{resolution}/left/{mode}/alpha_{str(alpha)}.p')
    results_paths = np.array(sorted(glob(results_regex)))
    for path in results_paths:
        mode = os.path.dirname(path).split("/")[-1]
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

                print(hemi, {n: round(np.nanmean(score), 4) for n, score in scores[hemi].items()})
                print(hemi, {f"{n}_max": round(np.nanmax(score), 2) for n, score in scores[hemi].items()})
                scores[hemi]["mean(imgs,captions)"] = scores[hemi]["overall"]
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
                    scores[hemi]['mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific)'] = np.array(
                        [np.mean((ai, ac)) - np.mean((si, sc)) for ai, ac, si, sc in
                         zip(scores[hemi]['images'], scores[hemi]['captions'],
                             scores_mod_specific_images['images'], scores_mod_specific_captions['captions'])])
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

                    scores[hemi]['imgs_agno - imgs_specific (cross)'] = np.array([ai - si for ai, ac, si, sc in
                                                                                  zip(scores[hemi]['images'],
                                                                                      scores[hemi]['captions'],
                                                                                      scores_mod_specific_captions[
                                                                                          'images'],
                                                                                      scores_mod_specific_images[
                                                                                          'captions'])])
                    scores[hemi]['captions_agno - captions_specific (cross)'] = np.array([ac - sc for ai, ac, si, sc in
                                                                                          zip(scores[hemi][
                                                                                                  'images'],
                                                                                              scores[hemi][
                                                                                                  'captions'],
                                                                                              scores_mod_specific_captions[
                                                                                                  'images'],
                                                                                              scores_mod_specific_images[
                                                                                                  'captions'])])
                    scores[hemi][
                        'mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific) (cross)'] = np.array(
                        [np.mean((ai, ac)) - np.mean((si, sc)) for ai, ac, si, sc in
                         zip(scores[hemi]['images'], scores[hemi]['captions'],
                             scores_mod_specific_captions['images'], scores_mod_specific_images['captions'])])
                    # scores[hemi]['imgs_specific (cross)'] = np.array([si for ai, ac, si, sc in
                    #                                                   zip(scores[hemi]['images'],
                    #                                                       scores[hemi]['captions'],
                    #                                                       scores_mod_specific_captions[
                    #                                                           'images'],
                    #                                                       scores_mod_specific_images[
                    #                                                           'captions'])])
                    # scores[hemi]['captions_specific (cross)'] = np.array([sc for ai, ac, si, sc in
                    #                                                       zip(scores[hemi]['images'],
                    #                                                           scores[hemi]['captions'],
                    #                                                           scores_mod_specific_captions[
                    #                                                               'images'],
                    #                                                           scores_mod_specific_images[
                    #                                                               'captions'])])

        add_to_all_scores(all_scores, scores)

        print("")

        all_subjects.add(subject)
        scores["subject"] = subject

        per_subject_scores.append(scores)

    # calc t-values
    for hemi in HEMIS:
        for score_name in all_scores[hemi].keys():
            all_scores[hemi][score_name] = [
                stats.ttest_1samp(x, popmean=CHANCE_VALUES[score_name], alternative="greater" if CHANCE_VALUES[score_name] == 0.5 else "two-sided") if (~np.isnan(x)).sum() == len(SUBJECTS) else np.nan for x
                in
                all_scores[hemi][score_name]]

    metrics = ["captions", "images", "mean(imgs,captions)", "min(imgs,captions)",
               'mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific)', 'imgs_agno - imgs_specific',
               'captions_agno - captions_specific'] #'imgs_agno - imgs_specific (cross)', 'captions_agno - captions_specific (cross)', 'mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific) (cross)'

    scores = all_scores
    fig = plt.figure(constrained_layout=True, figsize=(5 * len(VIEWS), len(metrics) * 2))
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
                    scores_hemi_t_values = np.array([d[0] if not np.isnan(d).any() else np.nan for d in scores_hemi])
                    infl_mesh = fsaverage[f"infl_{hemi}"]
                    if cbar_max is None:
                        cbar_max = min(np.nanmax(scores_hemi_t_values), 99)
                        cbar_min = np.nanmin(scores_hemi_t_values)

                    plotting.plot_surf_stat_map(
                        infl_mesh,
                        scores_hemi_t_values,
                        hemi=hemi,
                        view=view,
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        axes=axes[i * 2 + j],
                        colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                        threshold=3.365,    #p<0.01 for 5 degrees of freedom (6 subjects) (one-sided!)
                        vmax=99 if CHANCE_VALUES[metric] == 0.5 else None,
                        vmin=0.0 if CHANCE_VALUES[metric] == 0.5 else None,
                        cmap="hot" if CHANCE_VALUES[metric] == 0.5 else "cold_hot",
                        symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
                    )
                    axes[i * 2 + j].legend(
                        handles=[Circle((0, 0), radius=5, color='w', label=f"{hemi} {view}")], labelspacing=1,
                        borderpad=0, loc='upper center', frameon=False)  # bbox_to_anchor=(1.9, 0.8),
                else:
                    axes[i * 2 + j].axis('off')

    title = f"{model_name}_{mode}_group_level"
    fig.suptitle(title)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    # plt.subplots_adjust(hspace=0, wspace=0, right=0.85, left=0)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')

    # per-subject plots
    for scores in tqdm(per_subject_scores):
        fig = plt.figure(constrained_layout=True, figsize=(5 * len(VIEWS), len(metrics) * 2))
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
                        axes[i * 2 + j].legend(
                            handles=[Circle((0, 0), radius=5, color='w', label=f"{hemi} {view}")], labelspacing=1,
                            borderpad=0, loc='upper center', frameon=False)  # bbox_to_anchor=(1.9, 0.8),

                        # plotting.plot_surf_contours(infl_mesh, parcellation_surf, labels=labels,
                        #                             levels=regions_indices, axes=row_axes[i*2+j],
                        #                             legend=True,
                        #                             colors=colors)
                    else:
                        axes[i * 2 + j].axis('off')

        title = f"{model_name}_{mode}_{scores['subject']}"
        fig.suptitle(title)
        title += f"_alpha_{str(alpha)}"
        results_searchlight = os.path.join(RESULTS_DIR, "searchlight", resolution, f"{title}.png")
        os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
        plt.subplots_adjust(hspace=0, wspace=0, right=0.85, left=0)
        plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--subset", type=int, default=None)

    parser.add_argument("--models", type=str, nargs='+', default=['clip'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=VISION_MEAN_FEAT_KEY,
                        choices=VISION_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)
    parser.add_argument("--resolution", type=str, default="fsaverage")

    parser.add_argument("--l2-regularization-alpha", type=float, default=1e3)

    parser.add_argument("--radius", type=float, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
