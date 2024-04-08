import argparse
import warnings

import numpy as np
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle

from nilearn.surface import surface
from scipy import stats
from scipy.stats import pearsonr
from sklearn import neighbors
from tqdm import tqdm
import seaborn as sns

from utils import RESULTS_DIR

METRIC_CAPTIONS = 'captions'
METRIC_IMAGES = 'images'
METRIC_DIFF_CAPTIONS = 'captions_agno - captions_specific'
METRIC_DIFF_IMAGES = 'imgs_agno - imgs_specific'
METRIC_MIN_DIFF_BOTH_MODALITIES = 'min(captions_agno - captions_specific, imgs_agno - imgs_specific)'

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")

COLORBAR_MAX = 0.85
COLORBAR_THRESHOLD_MIN = 0.6
COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.02

DEFAULT_T_VALUE_THRESHOLD = 0.824
DEFAULT_MAX_CLUSTER_DISTANCE = 1  # 1mm
DEFAULT_MIN_CLUSTER_T_VALUE = 20 * DEFAULT_T_VALUE_THRESHOLD

VIEWS = ["lateral", "medial", "ventral"]

HEMIS = ['left', 'right']

BASE_METRICS = ["test_captions", "test_images"]
CHANCE_VALUES = {
    METRIC_CAPTIONS: 0.5,
    METRIC_IMAGES: 0.5,
    METRIC_DIFF_IMAGES: 0,
    METRIC_DIFF_CAPTIONS: 0,
    METRIC_MIN_DIFF_BOTH_MODALITIES: 0,
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


def process_scores(scores_agnostic, scores_captions, scores_images, nan_locations):
    scores = dict()
    for metric in BASE_METRICS:
        score_name = metric.split("_")[1]
        scores[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores[score_name][~nan_locations] = np.array([score[metric] for score in scores_agnostic])

    # correlation_num_voxels_acc(scores_agnostic, scores, hemi, nan_locations)

    scores_specific_captions = dict()
    for metric in BASE_METRICS:
        score_name = metric.split("_")[1]
        scores_specific_captions[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores_specific_captions[score_name][~nan_locations] = np.array(
            [score[metric] for score in scores_captions])

    scores_specific_images = dict()
    for metric in BASE_METRICS:
        score_name = metric.split("_")[1]
        scores_specific_images[score_name] = np.repeat(np.nan, nan_locations.shape)
        scores_specific_images[score_name][~nan_locations] = np.array(
            [score[metric] for score in scores_images])

    scores[METRIC_DIFF_IMAGES] = np.array(
        [ai - si for ai, ac, si, sc in
         zip(scores[METRIC_IMAGES],
             scores[METRIC_CAPTIONS],
             scores_specific_images[METRIC_IMAGES],
             scores_specific_captions[METRIC_CAPTIONS])]
    )
    scores[METRIC_DIFF_CAPTIONS] = np.array(
        [ac - sc for ai, ac, si, sc in
         zip(scores[METRIC_IMAGES],
             scores[METRIC_CAPTIONS],
             scores_specific_images[METRIC_IMAGES],
             scores_specific_captions[METRIC_CAPTIONS])]
    )

    return scores


def run(args):
    per_subject_scores = dict()
    all_scores = {hemi: dict() for hemi in HEMIS}

    all_scores_null_distr = []
    alpha = 1

    results_regex = os.path.join(SEARCHLIGHT_OUT_DIR,
                                 f'train/{args.model}/*/*/{args.resolution}/*/{args.mode}/alpha_{str(alpha)}.p')
    paths_mod_agnostic = np.array(sorted(glob(results_regex)))
    paths_mod_specific_captions = np.array(sorted(glob(results_regex.replace('train/', 'train_captions/'))))
    paths_mod_specific_images = np.array(sorted(glob(results_regex.replace('train/', 'train_images/'))))
    assert len(paths_mod_agnostic) == len(paths_mod_specific_images) == len(paths_mod_specific_captions)

    adjacency_matrices = dict()
    for hemi in HEMIS:
        adj_path = os.path.join(SEARCHLIGHT_OUT_DIR, "adjacency_matrices", args.resolution, "adjacency.p")
        if not os.path.isfile(adj_path):
            os.makedirs(os.path.dirname(adj_path), exist_ok=True)
            fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
            coords, _ = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])
            nn = neighbors.NearestNeighbors(radius=args.max_cluster_distance)
            adjacency = [np.argwhere(arr == 1)[:, 0] for arr in
                         nn.fit(coords).radius_neighbors_graph(coords).toarray()]
            adjacency_matrices[hemi] = adjacency
        else:
            adjacency_matrices[hemi] = pickle.load(open(adj_path, 'rb'))

    for path_agnostic, path_caps, path_imgs in zip(paths_mod_agnostic, paths_mod_specific_captions,
                                                   paths_mod_specific_images):
        print(path_agnostic)
        hemi = os.path.dirname(path_agnostic).split("/")[-2]
        subject = os.path.dirname(path_agnostic).split("/")[-4]

        results_agnostic = pickle.load(open(path_agnostic, 'rb'))
        scores_agnostic = results_agnostic['scores']
        scores_captions = pickle.load(open(path_caps, 'rb'))['scores']
        scores_images = pickle.load(open(path_imgs, 'rb'))['scores']

        nan_locations = results_agnostic['nan_locations']
        scores = process_scores(scores_agnostic, scores_captions, scores_images, nan_locations)
        print({n: round(np.nanmean(score), 4) for n, score in scores.items()})
        print({f"{n}_max": round(np.nanmax(score), 2) for n, score in scores.items()})
        add_to_all_scores(all_scores, scores, hemi)

        print("")

        if subject not in per_subject_scores.keys():
            per_subject_scores[subject] = dict()
        per_subject_scores[subject][hemi] = scores

        null_distribution_file_name = f"alpha_{str(alpha)}_null_distribution.p"
        null_distribution_agnostic = pickle.load(
            open(os.path.join(os.path.dirname(path_agnostic), null_distribution_file_name), 'rb'))

        null_distribution_images = pickle.load(
            open(os.path.join(os.path.dirname(path_imgs), null_distribution_file_name), 'rb'))

        null_distribution_captions = pickle.load(
            open(os.path.join(os.path.dirname(path_caps), null_distribution_file_name), 'rb'))

        for i, (distr, distr_caps, distr_imgs) in enumerate(zip(null_distribution_agnostic,
                                                                null_distribution_captions,
                                                                null_distribution_images)):
            if len(all_scores_null_distr) <= i:
                all_scores_null_distr.append({hemi: dict() for hemi in HEMIS})
            scores_null_distr = all_scores_null_distr[i]
            scores = process_scores(distr, distr_caps, distr_imgs, nan_locations)
            add_to_all_scores(scores_null_distr, scores, hemi)

    def calc_t_values(scores, num_subjects):
        t_values = {hemi: dict() for hemi in HEMIS}
        for hemi in HEMIS:
            for score_name in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]:
                data = scores[hemi][score_name]
                popmean = CHANCE_VALUES[score_name]
                enough_data = [(~np.isnan(x)).sum() == num_subjects for x in data]
                t_values[hemi][score_name] = np.array([
                    stats.ttest_1samp(x, popmean=popmean, alternative="greater")[0] if ed else np.nan for x, ed
                    in zip(data, enough_data)]
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
                    (t_values[hemi][METRIC_DIFF_CAPTIONS], t_values[hemi][METRIC_DIFF_IMAGES]),
                    axis=0)
        return t_values

    num_subjects = len(per_subject_scores)
    print(f"Calculating t-values for {num_subjects} subjects.")
    t_values = calc_t_values(all_scores, num_subjects)

    print(f"Calculating t-values for {num_subjects} subjects: null distribution")
    t_values_null_distribution = [calc_t_values(t_vals, num_subjects) for t_vals in tqdm(all_scores_null_distr)]

    def calc_clusters(t_values):
        clusters = {hemi: [] for hemi in HEMIS}
        for hemi in HEMIS:
            scores = t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES]
            scores_thresholded = scores > args.t_value_threshold
            adj = adjacency_matrices[hemi]

            start_locations = list(np.argwhere(scores_thresholded)[:, 0])
            while len(start_locations) > 0:
                idx = start_locations[0]
                cluster = {idx}
                checked = {idx}

                def expand_neighbors(start_idx):
                    neighbors = adj[start_idx]
                    for neighbor in neighbors:
                        if not neighbor in checked:
                            checked.add(neighbor)
                            if scores_thresholded[neighbor]:
                                cluster.add(neighbor)
                                expand_neighbors(neighbor)

                expand_neighbors(idx)
                for id in cluster:
                    start_locations.remove(id)
                t_value_cluster = np.sum(scores[list(cluster)])
                if t_value_cluster >= args.min_cluster_t_value:
                    clusters[hemi].append(cluster)
        return clusters


    clusters = calc_clusters(t_values)

    print(f"Calculating clusters for null distribution")
    clusters_null_distribution = [calc_clusters(t_vals) for t_vals in tqdm(t_values_null_distribution)]

    # for each location, calculate how often it's part of a cluster
    prob_part_of_cluster = {
        hemi: np.zeros_like(scores[METRIC_MIN_DIFF_BOTH_MODALITIES]) for hemi, scores in t_values.items()
    }
    for cluster_distr in clusters_null_distribution:
        for hemi in HEMIS:
            cluster_distr_hemi = cluster_distr[hemi]
            for cluster in cluster_distr_hemi:
                np.add.at(prob_part_of_cluster[hemi], list(cluster), 1)

    print(f"plotting (t-values) threshold {args.t_value_threshold}")
    metrics = [METRIC_DIFF_IMAGES,
               METRIC_DIFF_CAPTIONS,
               METRIC_MIN_DIFF_BOTH_MODALITIES]
    fig = plt.figure(figsize=(5 * len(VIEWS), len(metrics) * 2))
    subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    for subfig, metric in zip(subfigs, metrics):
        subfig.suptitle(f'{metric}', x=0, horizontalalignment="left")
        axes = subfig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
        cbar_max = None
        for i, view in enumerate(VIEWS):
            for j, hemi in enumerate(['left', 'right']):
                if metric in t_values[hemi].keys():
                    scores_hemi = t_values[hemi][metric]
                    infl_mesh = fsaverage[f"infl_{hemi}"]
                    if cbar_max is None:
                        cbar_max = min(np.nanmax(scores_hemi), 99)
                    plotting.plot_surf_stat_map(
                        infl_mesh,
                        scores_hemi,
                        hemi=hemi,
                        view=view,
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        axes=axes[i * 2 + j],
                        colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                        threshold=args.t_value_threshold if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else 2.015,
                        vmax=cbar_max,
                        vmin=0 if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else -cbar_max,
                        cmap="hot" if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else "cold_hot",
                        symmetric_cbar=False if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else True,
                    )
                    axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
                else:
                    axes[i * 2 + j].axis('off')
    title = f"{args.model}_{args.mode}_group_level_t_values_tresh_{args.t_value_threshold}"
    # fig.suptitle(title)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    metrics = [METRIC_CAPTIONS, METRIC_IMAGES,
               METRIC_DIFF_IMAGES,
               METRIC_DIFF_CAPTIONS]
    print(f"plotting group-level avg scores.")
    fig = plt.figure(figsize=(5 * len(VIEWS), len(metrics) * 2))
    subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    for subfig, metric in zip(subfigs, metrics):
        subfig.suptitle(f'{metric}', x=0, horizontalalignment="left")
        axes = subfig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
        cbar_max = None
        for i, view in enumerate(VIEWS):
            for j, hemi in enumerate(['left', 'right']):
                if metric in all_scores[hemi].keys():
                    scores_hemi = all_scores[hemi][metric]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        scores_hemi_averaged = np.nanmean(scores_hemi, axis=1)
                    infl_mesh = fsaverage[f"infl_{hemi}"]
                    if cbar_max is None:
                        cbar_max = min(np.nanmax(scores_hemi_averaged), 99)

                    plotting.plot_surf_stat_map(
                        infl_mesh,
                        scores_hemi_averaged,
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
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    if args.per_subject_plots:
        # per-subject plots
        metrics = [METRIC_CAPTIONS, METRIC_IMAGES,
                   METRIC_DIFF_IMAGES,
                   METRIC_DIFF_CAPTIONS]
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

    parser.add_argument("--t-value-threshold", type=float, default=DEFAULT_T_VALUE_THRESHOLD)
    parser.add_argument("--max-cluster-distance", type=float, default=DEFAULT_MAX_CLUSTER_DISTANCE)
    parser.add_argument("--min-cluster-t-value", type=int, default=DEFAULT_MIN_CLUSTER_T_VALUE)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
