import argparse
import copy
import warnings
import random

import numpy as np
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle

from nilearn.experimental.surface import SurfaceMasker, SurfaceImage, load_fsaverage
from nilearn.surface import surface, load_surface
from nilearn.mass_univariate._utils import calculate_tfce
from scipy import stats
from scipy.stats import pearsonr, false_discovery_control
from sklearn import neighbors
from tqdm import tqdm
import seaborn as sns

from analyses.searchlight import SEARCHLIGHT_OUT_DIR
from utils import RESULTS_DIR, SUBJECTS, HEMIS

METRIC_CAPTIONS = 'captions'
METRIC_IMAGES = 'images'
METRIC_DIFF_CAPTIONS = 'captions_agno - captions_specific'
METRIC_DIFF_IMAGES = 'imgs_agno - imgs_specific'
METRIC_MIN_DIFF_BOTH_MODALITIES = 'min(captions_agno - captions_specific, imgs_agno - imgs_specific)'

COLORBAR_MAX = 0.85
COLORBAR_THRESHOLD_MIN = 0.6
COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.02

DEFAULT_T_VALUE_THRESHOLD = 0.824
DEFAULT_MAX_CLUSTER_DISTANCE = 1  # 1mm
DEFAULT_MIN_CLUSTER_T_VALUE = 20 * DEFAULT_T_VALUE_THRESHOLD
DEFAULT_CLUSTER_SIZE = 10

VIEWS = ["lateral", "medial", "ventral"]

BASE_METRICS = ["test_captions", "test_images"]
TEST_METRICS = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES]
CHANCE_VALUES = {
    METRIC_CAPTIONS: 0.5,
    METRIC_IMAGES: 0.5,
    METRIC_DIFF_IMAGES: 0,
    METRIC_DIFF_CAPTIONS: 0,
    METRIC_MIN_DIFF_BOTH_MODALITIES: 0,
}


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


def get_adj_matrices(resolution, max_cluster_distance):
    adj_path = os.path.join(SEARCHLIGHT_OUT_DIR, "adjacency_matrices", resolution,
                            f"adjacency_max_dist_{max_cluster_distance}.p")
    if not os.path.isfile(adj_path):
        adjacency_matrices = dict()
        for hemi in HEMIS:
            os.makedirs(os.path.dirname(adj_path), exist_ok=True)
            fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
            coords, _ = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])
            nn = neighbors.NearestNeighbors(radius=max_cluster_distance)
            adjacency = [np.argwhere(arr == 1)[:, 0] for arr in
                         nn.fit(coords).radius_neighbors_graph(coords).toarray()]
            adjacency_matrices[hemi] = adjacency
        pickle.dump(adjacency_matrices, open(adj_path, "wb"))
    else:
        adjacency_matrices = pickle.load(open(adj_path, 'rb'))

    # adjacency matrices for fixed cluster size:
    # adjacency_matrices = dict()
    # for hemi in HEMIS:
    #     adj_path = os.path.join(SEARCHLIGHT_OUT_DIR, "adjacency_matrices", args.resolution, f"adjacency_cluster_size_{args.cluster_size}.p")
    #     if not os.path.isfile(adj_path):
    #         print("creating adjacency matrices for cluster size", args.cluster_size)
    #         os.makedirs(os.path.dirname(adj_path), exist_ok=True)
    #         fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    #         coords, _ = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])
    #         nn = neighbors.NearestNeighbors()
    #         distances, adjacency = nn.fit(coords).kneighbors(coords, n_neighbors=args.cluster_size-1)
    #         adjacency_matrices[hemi] = adjacency

    return adjacency_matrices


def calc_clusters_variable_size(t_values, adjacency_matrices, t_value_threshold, return_clusters=False):
    clusters = {hemi: [] for hemi in HEMIS}
    cluster_t_values = {hemi: [] for hemi in HEMIS}
    cluster_maps = dict()
    for hemi in HEMIS:
        scores = t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES]
        scores_thresholded = scores > t_value_threshold
        adj = adjacency_matrices[hemi]

        start_locations = list(np.argwhere(scores_thresholded)[:, 0])
        cluster_maps[hemi] = np.zeros_like(scores)
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
            t_value_cluster = np.sum(scores[list(cluster)])     #- t_value_threshold)
            cluster_t_values[hemi].append(t_value_cluster)
            clusters[hemi].append(cluster)
            cluster_maps[hemi][list(cluster)] = scores[list(cluster)]

        # fill non-cluster locations with zeros
        cluster_maps[hemi][cluster_maps[hemi] == 0] = 0

    cluster_sizes = [len(c) for hemi in HEMIS for c in clusters[hemi]]
    max_cluster_size = np.max(cluster_sizes) if len(cluster_sizes) > 0 else 0
    if return_clusters:
        return clusters, cluster_maps, max_cluster_size, cluster_t_values
    else:
        return max_cluster_size, cluster_t_values


def calc_t_values(per_subject_scores):
    t_values = {hemi: dict() for hemi in HEMIS}
    for hemi in HEMIS:
        t_vals = dict()
        for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]:
            data = np.array([per_subject_scores[subj][hemi][metric] for subj in SUBJECTS])
            popmean = CHANCE_VALUES[metric]
            enough_data = np.isnan(data).sum(axis=0) == 0
            t_vals[metric] = np.array([
                stats.ttest_1samp(x, popmean=popmean, alternative="greater")[0] if ed else np.nan for x, ed
                in zip(data.T, enough_data)]
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
                (t_vals[METRIC_DIFF_CAPTIONS], t_vals[METRIC_DIFF_IMAGES]),
                axis=0)
    return t_values


def run(args):
    per_subject_scores = {subj: dict() for subj in SUBJECTS}
    alpha = 1
    features = "concat"

    results_regex = os.path.join(SEARCHLIGHT_OUT_DIR,
                                 f'train/{args.model}/{features}/*/{args.resolution}/*/{args.mode}/alpha_{str(alpha)}.p')
    paths_mod_agnostic = np.array(sorted(glob(results_regex)))
    paths_mod_specific_captions = np.array(sorted(glob(results_regex.replace('train/', 'train_captions/'))))
    paths_mod_specific_images = np.array(sorted(glob(results_regex.replace('train/', 'train_images/'))))
    assert len(paths_mod_agnostic) == len(paths_mod_specific_images) == len(paths_mod_specific_captions)

    adjacency_matrices = get_adj_matrices(args.resolution, args.max_cluster_distance)

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

        per_subject_scores[subject][hemi] = scores
        print("")

    t_values_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, features, args.resolution, args.mode,
                                 "t_values.p")
    if not os.path.isfile(t_values_path):
        os.makedirs(os.path.dirname(t_values_path), exist_ok=True)
        print(f"Calculating t-values")
        t_values = calc_t_values(per_subject_scores)

        pickle.dump(t_values, open(t_values_path, 'wb'))
    else:
        t_values = pickle.load(open(t_values_path, 'rb'))

    # data = t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES]
    # fsaverage = load_fsaverage(mesh_name=args.resolution)
    # # pial_mesh = fsaverage[f"pial_{hemi}"]
    #
    # # surf_data = load_surface((pial_mesh, data))
    # # mesh = {
    # #     "left": fsaverage[f"pial_left"],
    # #     "right": fsaverage[f"pial_right"],
    # # }
    # data = {
    #     "left_hemisphere": t_values["left"][METRIC_MIN_DIFF_BOTH_MODALITIES],
    #     "right_hemisphere": t_values["right"][METRIC_MIN_DIFF_BOTH_MODALITIES],
    # }
    # surf_img = SurfaceImage(mesh=fsaverage["pial"], data=data)
    #
    # masker = SurfaceMasker()
    # masked_data = masker.fit_transform(surf_img)
    # print(f"Masked data shape: {masked_data.shape}")
    # # masker.inverse_transform(surf_img)

    # avg_values = {hemi: dict() for hemi in HEMIS}
    # for hemi in HEMIS:
    #     for metric in TEST_METRICS:
    #         avg_values[hemi][metric] = np.mean([ps[hemi][metric] for ps in per_subject_scores.values()], axis=0)
    #
    #     avg_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
    #         (avg_values[hemi][METRIC_DIFF_CAPTIONS], avg_values[hemi][METRIC_DIFF_IMAGES]),
    #         axis=0)

    # def create_avg_maps_null_distr(per_subject_scores, n_iter=100000):
    #     all_avg_maps = []
    #     for _ in tqdm(range(n_iter)):
    #         avg_maps = {hemi: dict() for hemi in HEMIS}
    #         for hemi in HEMIS:
    #             for score_name in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]:
    #                 random_idx = np.random.choice(len(per_subject_scores), size=len(SUBJECTS))
    #                 data = np.array(
    #                     [per_subject_scores[idx][subj][hemi][score_name] for idx, subj in zip(random_idx, SUBJECTS)]).T
    #                 # group_maps[hemi][score_name] = data
    #                 avg_maps[hemi][score_name] = data.mean(axis=1)
    #
    #             with warnings.catch_warnings():
    #                 warnings.simplefilter("ignore", category=RuntimeWarning)
    #                 avg_maps[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
    #                     (avg_maps[hemi][METRIC_DIFF_CAPTIONS], avg_maps[hemi][METRIC_DIFF_IMAGES]),
    #                     axis=0)
    #         all_avg_maps.append(avg_maps)
    #     return all_avg_maps
    #
    # print("calculating avg values")
    # avg_maps_null_distribution = create_avg_maps_null_distr(all_scores_null_distr)
    #
    # avg_maps_null_distribution_thresholded = [{hemi: dict() for hemi in HEMIS} for _ in
    #                                           range(len(avg_maps_null_distribution))]
    #
    # avg_values_thresholded = {hemi: dict() for hemi in HEMIS}
    # for hemi in HEMIS:
    #     mean_values = np.stack([avg_maps_null_distribution[i][hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] for i in
    #                             range(len(avg_maps_null_distribution))])
    #     thresholds = [sorted(np.abs(v))[-100] for v in mean_values.T]
    #     avg_values_thresholded[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.abs(
    #         avg_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES]) > thresholds
    #
    #     for map, map_thresholded in zip(avg_maps_null_distribution, avg_maps_null_distribution_thresholded):
    #         map_thresholded[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.abs(
    #             map[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES]) > thresholds


    # def calc_clusters_fixed_size(t_values):
    #     cluster_maps = dict()
    #     for hemi in HEMIS:
    #         scores = t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES]
    #         scores_thresholded = scores.copy()
    #         scores_thresholded[scores < args.t_value_threshold] = 0
    #         adj = adjacency_matrices[hemi]
    #
    #         t_values_cluster = np.array([np.nansum(scores[a]) for a in adj])
    #         t_values_cluster_thresholded = t_values_cluster.copy()
    #         t_values_cluster_thresholded[t_values_cluster < args.min_cluster_t_value] = 0
    #         cluster_maps[hemi] = t_values_cluster
    #     return cluster_maps

    print("calculating clusters..")
    clusters, cluster_maps, max_cluster_size, cluster_t_values = calc_clusters_variable_size(t_values,
                                                                                             adjacency_matrices,
                                                                                             args.t_value_threshold,
                                                                                             return_clusters=True)

    filename = f"clusters_null_distribution_t_thresh_{args.t_value_threshold}_max_dist_{args.max_cluster_distance}.p"
    clusters_null_distribution_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, features,
                                                   args.resolution,
                                                   args.mode, filename)
    clusters_null_distribution = pickle.load(open(clusters_null_distribution_path, 'rb'))

    max_cluster_size_distr = sorted([max_size for max_size, _ in clusters_null_distribution])
    significance_cutoff_size = np.quantile(max_cluster_size_distr, 0.95)
    print("cluster size significance cutoff for p<0.05: ", significance_cutoff_size)

    max_cluster_t_value_distr = sorted([
        np.max(c['left'] + c['right']) if len(c['left'] + c['right']) > 0 else 0 for _, c in clusters_null_distribution
    ])
    # mean_cluster_t_value_distr = sorted([
    #     np.mean(c['left'] + c['right']) if len(c['left'] + c['right']) > 0 else 0 for _, c in clusters_null_distribution
    # ])
    # max_cluster_t_value_distr = np.concatenate([c['left'] + c['right'] for _, c in clusters_null_distribution])
    significance_cutoff = np.quantile(max_cluster_t_value_distr, 0.95)
    print(f"cluster t-value significance cutoff for p<0.05 ({len(clusters_null_distribution)} permutations): {significance_cutoff}")

    p_values_cluster = copy.deepcopy(cluster_maps)
    for hemi in HEMIS:
        print(f"{hemi} hemi largest cluster sizes: ", sorted([len(cluster) for cluster in clusters[hemi]], reverse=True)[:10])
        print(f"{hemi} hemi largest cluster t-values: ", sorted([t for t in cluster_t_values[hemi]], reverse=True)[:10])
        for cluster, t_val in zip(clusters[hemi], cluster_t_values[hemi]):
            value_indices = np.argwhere(max_cluster_t_value_distr > t_val)
            p_value = 1 - value_indices[0] / len(clusters_null_distribution) if len(value_indices) > 0 else 1 - (len(clusters_null_distribution) - 1) / (len(clusters_null_distribution))
            p_values_cluster[hemi][list(cluster)] = p_value

    # FDR correction:
    # p_values_left = p_values_cluster['left'][p_values_cluster['left'] > 0]
    # p_values_right = p_values_cluster['right'][p_values_cluster['right'] > 0]
    # all_p_values = np.concatenate((p_values_left, p_values_right))
    # all_p_values_corrected = false_discovery_control(all_p_values, method='bh')
    # p_values_cluster['left'][p_values_cluster['left'] > 0] = all_p_values_corrected[:len(p_values_left)]
    # p_values_cluster['right'][p_values_cluster['right'] > 0] = all_p_values_corrected[len(p_values_left):]

    # transform to plottable magnitudes:
    p_values_cluster['left'][p_values_cluster['left'] > 0] = -np.log10(p_values_cluster['left'][p_values_cluster['left'] > 0])
    p_values_cluster['right'][p_values_cluster['right'] > 0] = -np.log10(p_values_cluster['right'][p_values_cluster['right'] > 0])

    print(f"plotting (p-values)")
    metric = METRIC_MIN_DIFF_BOTH_MODALITIES
    fig = plt.figure(figsize=(5 * len(VIEWS), 2))
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    fig.suptitle(f'{metric}', x=0, horizontalalignment="left")
    axes = fig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
    cbar_max = None
    for i, view in enumerate(VIEWS):
        for j, hemi in enumerate(HEMIS):
            scores_hemi = p_values_cluster[hemi]
            infl_mesh = fsaverage[f"infl_{hemi}"]
            if cbar_max is None:
                cbar_max = min(np.nanmax(scores_hemi), 99)
                print(cbar_max)
            plotting.plot_surf_stat_map(
                infl_mesh,
                scores_hemi,
                hemi=hemi,
                view=view,
                bg_map=fsaverage[f"sulc_{hemi}"],
                axes=axes[i * 2 + j],
                colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                threshold=1.3,  # -log10(0.05) ~ 1.3
                vmax=cbar_max,
                vmin=0,
                cmap="bwr",
                symmetric_cbar=False,
            )
            axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
    title = f"{args.model}_{args.mode}_group_level_-log10_p_values"
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"plotting (cluster t-values)")
    metric = METRIC_MIN_DIFF_BOTH_MODALITIES
    fig = plt.figure(figsize=(5 * len(VIEWS), 2))
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    fig.suptitle(f'{metric}', x=0, horizontalalignment="left")
    axes = fig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
    cbar_max = None
    for i, view in enumerate(VIEWS):
        for j, hemi in enumerate(HEMIS):
            scores_hemi = cluster_maps[hemi]
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
                threshold=args.t_value_threshold,
                vmax=cbar_max,
                vmin=0,
                cmap="hot",
                symmetric_cbar=False,
            )
            axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
    title = f"{args.model}_{args.mode}_group_level_cluster_t_values"
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"plotting (t-values) threshold {args.t_value_threshold}")
    metrics = [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, METRIC_MIN_DIFF_BOTH_MODALITIES]
    fig = plt.figure(figsize=(5 * len(VIEWS), len(metrics) * 2))
    subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    for subfig, metric in zip(subfigs, metrics):
        subfig.suptitle(f'{metric}', x=0, horizontalalignment="left")
        axes = subfig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
        cbar_max = None
        for i, view in enumerate(VIEWS):
            for j, hemi in enumerate(HEMIS):
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

    metrics = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]
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
    title = f"{args.model}_{args.mode}_group_level_pairwise_acc"
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    title += f"_alpha_{str(alpha)}"
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

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
    # parser.add_argument("--min-cluster-t-value", type=int, default=DEFAULT_MIN_CLUSTER_T_VALUE)

    # parser.add_argument("--cluster-size", type=int, default=DEFAULT_CLUSTER_SIZE)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
