import argparse
import copy
import itertools
import math

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn import datasets
import os
from glob import glob
import pickle

from nilearn.surface import surface
from scipy.spatial.distance import cdist
from tqdm import tqdm, trange

from analyses.decoding.searchlight.searchlight_classification import SEARCHLIGHT_CLASSIFICATION_PERMUTATION_TESTING_RESULTS_DIR, \
    searchlight_mode_from_args, get_results_file_path, get_adjacency_matrix
from data import MODALITY_AGNOSTIC, MODALITY_SPECIFIC_IMAGES, MODALITY_SPECIFIC_CAPTIONS, SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, LatentFeatsConfig, VISION_FEAT_COMBINATION_CHOICES, LANG_FEAT_COMBINATION_CHOICES, \
    TRAINING_MODES, TEST_IMAGES, TEST_CAPTIONS, TEST_CAPTIONS_ATTENDED, TEST_IMAGES_ATTENDED, TEST_IMAGES_UNATTENDED, \
    TEST_CAPTIONS_UNATTENDED, SPLIT_IMAGERY_WEAK, ATTENTION_MOD_SPLITS
from eval import LIMITED_CANDIDATE_LATENTS
from utils import SUBJECTS_ADDITIONAL_TEST, HEMIS, DEFAULT_RESOLUTION, DATA_DIR, DEFAULT_MODEL, FS_HEMI_NAMES, \
    export_to_gifti, DIFF_DECODERS, METRIC_MOD_INVARIANT_ATTENDED, \
    METRIC_MOD_INVARIANT_UNATTENDED, METRIC_MOD_INVARIANT_ATTENDED_ALT, \
    METRIC_MOD_INVARIANT_UNATTENDED_ALT, METRIC_ATTENTION_DIFF_CAPTIONS, METRIC_ATTENTION_DIFF_IMAGES, \
    METRIC_MOD_INVARIANT_INCREASE, DIFF

DEFAULT_N_JOBS = 10


DEFAULT_P_VAL_THRESHOLD = 1e-3


def calc_clusters(scores, threshold, edge_lengths=None, return_clusters=True,
                  return_cluster_edge_lengths=False, return_agg_t_values=False,
                  return_cluster_map=False):
    cluster_nodes = dict()
    cluster_edge_lengths = dict()

    # Filter edges for edges that are connecting nodes with score above threshold
    edge_lengths = {
        e: l for e, l in edge_lengths.items() if (scores[e[0]] >= threshold) and (scores[e[1]] >= threshold)
    }

    node_to_cluster = dict()

    next_cluster_id = 0
    for (n0, n1), length in edge_lengths.items():
        if n0 in node_to_cluster.keys() or n1 in node_to_cluster.keys():
            if n0 in node_to_cluster.keys() and n1 in node_to_cluster.keys():
                cluster_1_id, cluster_2_id = sorted([node_to_cluster[n0], node_to_cluster[n1]])
                if cluster_1_id == cluster_2_id:
                    cluster_edge_lengths[cluster_1_id] += length
                    continue

                # merge 2 clusters
                for node in cluster_nodes[cluster_2_id]:
                    node_to_cluster[node] = cluster_1_id
                cluster_nodes[cluster_1_id] = cluster_nodes[cluster_1_id] | cluster_nodes[cluster_2_id]
                cluster_edge_lengths[cluster_1_id] += cluster_edge_lengths[cluster_2_id]
                del cluster_nodes[cluster_2_id]
                del cluster_edge_lengths[cluster_2_id]
                continue

            elif n0 in node_to_cluster.keys():
                cluster_id = node_to_cluster[n0]
            else:
                cluster_id = node_to_cluster[n1]
        else:
            cluster_id = next_cluster_id
            next_cluster_id = next_cluster_id + 1
            cluster_nodes[cluster_id] = set()
            cluster_edge_lengths[cluster_id] = 0

        node_to_cluster[n0] = cluster_id
        node_to_cluster[n1] = cluster_id
        cluster_nodes[cluster_id] = cluster_nodes[cluster_id] | {n0, n1}
        cluster_edge_lengths[cluster_id] += length

    result_dict = dict()
    if return_clusters:
        result_dict['clusters'] = list(cluster_nodes.values())
    if return_cluster_map:
        cluster_map = np.zeros_like(scores)
        for cluster in cluster_nodes.values():
            cluster_t_value = sum(scores[n] for n in cluster)
            cluster_map[list(cluster)] = cluster_t_value
        result_dict['cluster_map'] = cluster_map
    if return_agg_t_values:
        cluster_t_values = [sum(scores[n] for n in cluster) for cluster in cluster_nodes.values()]
        result_dict['agg_t_values'] = cluster_t_values
    if return_cluster_edge_lengths:
        result_dict['cluster_edge_lengths'] = list(cluster_edge_lengths.values())
    return result_dict


def create_results_cluster_masks(values, results_dir, metric, resolution, radius, n_neighbors,
                                 threshold):
    tfce_values_path = os.path.join(results_dir, f"tfce_values_{metric}.p")
    tfce_values = pickle.load(open(tfce_values_path, "rb"))

    p_values_path = os.path.join(results_dir, f"p_values_{metric}.p")
    p_values = pickle.load(open(p_values_path, "rb"))

    edge_lengths = get_edge_lengths_dicts_based_on_edges(resolution)
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")

    results_maps_path = os.path.join(results_dir, f"results_maps")
    masks_path = os.path.join(os.path.dirname(p_values_path), f"masks_{metric}")
    os.makedirs(masks_path, exist_ok=True)

    clusters_df = []
    for hemi in HEMIS:
        print(f"\nclusters for {hemi} hemi")

        adj = None
        if radius is not None or n_neighbors is not None:
            adj, _, _ = get_adjacency_matrix(hemi, resolution, radius=radius, num_neighbors=n_neighbors)

        mesh = surface.load_surf_mesh(fsaverage[f"white_{hemi}"])
        results = calc_clusters(values[hemi], threshold=1e-8, edge_lengths=edge_lengths[hemi], return_clusters=True)
        clusters = results['clusters']
        clusters.sort(key=len, reverse=True)
        for i, cluster in enumerate(clusters[:10]):
            cluster = list(cluster)
            print(f"Cluster {i}: {len(cluster)} vertices", end=" | ")
            vertex_max_tfce_value = cluster[np.nanargmax(tfce_values[hemi][metric][cluster])]
            max_tfce_value = tfce_values[hemi][metric][vertex_max_tfce_value]
            print(f"Max tfce-value: {max_tfce_value:.2f}", end=" | ")
            coords = mesh.coordinates[vertex_max_tfce_value]
            print(f"Coordinates (max t-value): {coords}")
            clusters_df.append({
                "hemi": hemi, "id": i, "location": "", "size": len(cluster),
                "max tfce-value": vertex_max_tfce_value,
                "p-value": '{:.0e}'.format(p_values[hemi][vertex_max_tfce_value]),
                "peak coordinates": np.round(coords, 1),
                "references": ""
            })

            cluster_map = np.repeat(np.nan, p_values[hemi].shape)
            cluster_map[list(cluster)] = values[hemi][cluster]
            fname = f"{metric}_{FS_HEMI_NAMES[hemi]}_threshold_{threshold}_cluster_{i}.gii"
            path_out = os.path.join(results_maps_path, f"clusters", fname)
            os.makedirs(os.path.dirname(path_out), exist_ok=True)
            export_to_gifti(cluster_map, path_out)

            path_out = os.path.join(results_maps_path, f"cluster_masks", fname.replace(".gii", ".p"))
            os.makedirs(os.path.dirname(path_out), exist_ok=True)
            mask = {hemi: np.repeat(np.nan, p_values[hemi].shape) for hemi in HEMIS}
            mask[hemi] = cluster_map
            pickle.dump(mask, open(path_out, "wb"))

            if adj is not None:
                cluster_map_extended = np.repeat(np.nan, p_values[hemi].shape)
                cluster_map_extended[np.unique([adj[cluster_idx] for cluster_idx in cluster])] = 1
                fname = f"{metric}_{FS_HEMI_NAMES[hemi]}_threshold_{threshold}_cluster_{i}.gii"
                path_out = os.path.join(results_maps_path, f"clusters_extended", fname)
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                export_to_gifti(cluster_map_extended, path_out)

                path_out = os.path.join(results_maps_path, f"cluster_extended_masks", fname.replace(".gii", ".p"))
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                mask = {hemi: np.repeat(np.nan, p_values[hemi].shape) for hemi in HEMIS}
                mask[hemi] = cluster_map_extended
                pickle.dump(mask, open(path_out, "wb"))

    df = pd.DataFrame.from_records(clusters_df, index=["hemi", "id"])
    print(df.style.format(precision=3).to_latex(hrules=True))


def calc_significance_cutoff(null_distribution_tfce_values, metric, p_value_threshold=0.05):
    null_distr = np.sort([
        np.nanmax(np.concatenate((n[HEMIS[0]][metric], n[HEMIS[1]][metric])))
        for n in null_distribution_tfce_values
    ])
    print(f"null distr max values: {null_distr[-5:]} ({len(null_distribution_tfce_values)} permutations)")

    if p_value_threshold == 1 / len(null_distribution_tfce_values):
        significance_cutoff = np.max(null_distr)
    else:
        significance_cutoff = np.quantile(null_distr, 1 - p_value_threshold, method='closest_observation')

    for thresh in [0.05, 1e-2, 1e-3, 1e-4]:
        if thresh == 1 / len(null_distribution_tfce_values):
            val = np.max(null_distr)
        else:
            val = np.quantile(null_distr, 1 - thresh, method='closest_observation')
        print(f"(info) cluster test statistic significance cutoff for p<{thresh}: {val:.2f}")

    print(f"using cluster test statistic significance cutoff for p<{p_value_threshold}: {significance_cutoff:.3f}")

    return significance_cutoff, null_distr


def create_masks(results_dir, metric, significance_cutoff, tfce_value_threshold, resolution, radius=None,
                 n_neighbors=None):
    print("Creating gifti results masks")
    p_values_path = os.path.join(results_dir, f"p_values_{metric}.p")

    results_maps_path = os.path.join(results_dir, "results_maps")
    os.makedirs(results_maps_path, exist_ok=True)

    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    log_10_p_values = copy.deepcopy(p_values)
    log_10_p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    log_10_p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    for hemi in HEMIS:
        path_out = os.path.join(results_maps_path, f"p_values_{metric}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(log_10_p_values[hemi], path_out)

    tfce_values_path = os.path.join(results_dir, f"tfce_values_{metric}.p")
    tfce_values = pickle.load(open(tfce_values_path, "rb"))

    threshold = significance_cutoff
    if tfce_value_threshold is not None:
        threshold = tfce_value_threshold
        print(f"using tfce value threshold {tfce_value_threshold}")
    masks = {hemi: copy.deepcopy(tfce_values[hemi][metric]) for hemi in HEMIS}
    for hemi in HEMIS:
        print(
            f'{hemi} hemi mask size for threshold {threshold:.2f}: {np.mean(tfce_values[hemi][metric] >= threshold):.2f}')
        masks[hemi][tfce_values[hemi][metric] >= threshold] = 1
        masks[hemi][tfce_values[hemi][metric] < threshold] = 0
        masks[hemi][np.isnan(tfce_values[hemi][metric])] = 0
        masks[hemi] = masks[hemi].astype(np.uint8)

        path_out = os.path.join(results_maps_path, f"tfce_values_{metric}_{FS_HEMI_NAMES[hemi]}.gii")
        tfce_values[hemi][metric][tfce_values[hemi][metric] < threshold] = 0
        export_to_gifti(tfce_values[hemi][metric], path_out)

    # create_results_cluster_masks(masks, results_dir, metric, resolution, radius, n_neighbors, threshold)


def get_edge_lengths_dicts_based_on_edges(resolution):
    edge_lengths_dicts = dict()
    fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
    surface_infl = {hemi: surface.load_surf_mesh(fsaverage[f"infl_{hemi}"]) for hemi in HEMIS}
    for hemi in HEMIS:
        coords = surface_infl[hemi].coordinates

        edges = np.vstack([surface_infl[hemi].faces[:, [0, 1]],
                           surface_infl[hemi].faces[:, [0, 2]],
                           surface_infl[hemi].faces[:, [1, 2]]])
        edges = np.array([(e0, e1) if e0 < e1 else (e1, e0) for e0, e1 in edges])
        lengths = np.sqrt(np.sum((coords[edges[:, 0]] - coords[edges[:, 1]]) ** 2, axis=1))
        edge_lengths_dicts[hemi] = {(e[0], e[1]): l for e, l in zip(edges, lengths)}

    return edge_lengths_dicts


def compute_composite_t_vals_for_metric(t_values, metric, hemi):
    if metric in T_VAL_METRICS:
        values = t_values[hemi][metric]
    # elif metric == METRIC_GW:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_GW], axis=0)
    # elif metric == METRIC_GW_2:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_GW_2], axis=0)
    # elif metric == METRIC_GW_3:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_GW_3], axis=0)
    # elif metric == METRIC_GW_4:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_GW_4], axis=0)
    # elif metric == METRIC_GW_5:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_GW_5], axis=0)
    elif metric == METRIC_MOD_INVARIANT_INCREASE:
        values_attn = np.nanmin(
            [
                t_values[hemi]['$'.join([MODALITY_SPECIFIC_IMAGES, TEST_CAPTIONS_ATTENDED])],
                t_values[hemi]['$'.join([MODALITY_SPECIFIC_CAPTIONS, TEST_IMAGES_ATTENDED])]
            ], axis=0
        )
        values_unattn = np.nanmin(
            [
                t_values[hemi]['$'.join([MODALITY_SPECIFIC_IMAGES, TEST_CAPTIONS_UNATTENDED])],
                t_values[hemi]['$'.join([MODALITY_SPECIFIC_CAPTIONS, TEST_IMAGES_UNATTENDED])]
            ], axis=0
        )
        values_attn[values_attn < 0] = 0
        values_unattn[values_unattn < 0] = 0
        values = values_attn - values_unattn
    elif metric == METRIC_ATTENTION_DIFF_IMAGES:
        values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_ATTENTION_DIFF_IMAGES], axis=0)
    elif metric == METRIC_ATTENTION_DIFF_CAPTIONS:
        values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_ATTENTION_DIFF_CAPTIONS], axis=0)
    elif metric == METRIC_MOD_INVARIANT_ATTENDED:
        values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_MOD_INVARIANT_ATTENDED], axis=0)
    elif metric == METRIC_MOD_INVARIANT_UNATTENDED:
        values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_MOD_INVARIANT_UNATTENDED], axis=0)
    elif metric == METRIC_MOD_INVARIANT_ATTENDED_ALT:
        values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_MOD_INVARIANT_ATTENDED_ALT], axis=0)
    elif metric == METRIC_MOD_INVARIANT_UNATTENDED_ALT:
        values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_MOD_INVARIANT_UNATTENDED_ALT], axis=0)
    # elif metric == METRIC_GW_DIFF:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_GW_DIFF], axis=0)
    # elif metric == METRIC_VISION:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_VISION], axis=0)
    # elif metric == METRIC_LANG:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_LANG], axis=0)
    # elif metric == METRIC_VISION_2:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_VISION_2], axis=0)
    # elif metric == METRIC_LANG_2:
    #     values = np.nanmin([t_values[hemi][m] for m in T_VAL_METRICS_LANG_2], axis=0)
    else:
        raise RuntimeError("Unknown metric: ", metric)
    return values


def calc_tfce_values(t_values, edge_lengths_dicts, metric, h=2, e=1, dh=0.1, cluster_extents_measure="num_vertices",
                     use_tqdm=False):
    tfce_values = dict()
    for hemi in HEMIS:
        values = compute_composite_t_vals_for_metric(t_values, metric, hemi)
        max_score = np.nanmax(values)
        if max_score <= 0:
            tfce_values[hemi] = {metric: np.zeros_like(values)}
            continue

        if dh == 0:
            dh = "auto"

        if dh == "auto":
            step = max_score / 100
            if use_tqdm:
                print(f"Automatically set dh to {step}")
        else:
            step = dh

        score_threshs = np.arange(step, max_score + step, step)

        tfce_values[hemi] = {metric: np.zeros(shape=values.shape, dtype=np.float32)}

        iterator = tqdm(score_threshs) if use_tqdm else score_threshs
        for score_thresh in iterator:
            clusters_dict = calc_clusters(
                values,
                score_thresh,
                edge_lengths_dicts[hemi],
                return_clusters=True,
                return_cluster_edge_lengths=True,
            )
            clusters = clusters_dict["clusters"]
            if cluster_extents_measure == "num_vertices":
                cluster_extents = np.array([len(c) for c in clusters])
            elif cluster_extents_measure == "edge_lengths":
                cluster_extents = np.array(clusters_dict["cluster_edge_lengths"])
            else:
                raise RuntimeError('Unknown cluster extents measure: ', cluster_extents_measure)

            cluster_tfces = (cluster_extents ** e) * (score_thresh ** h) * step
            nodes_above_thresh_not_in_clusters = set(np.argwhere(values > score_thresh)[:, 0])
            for cluster, cluster_tfce in zip(clusters, cluster_tfces):
                tfce_values[hemi][metric][list(cluster)] += cluster_tfce
                nodes_above_thresh_not_in_clusters = nodes_above_thresh_not_in_clusters.difference(cluster)

            # increase tfce values for nodes out of clusters
            if cluster_extents_measure == "num_vertices":
                if len(nodes_above_thresh_not_in_clusters) > 0:
                    single_node_tfce = (1 ** e) * (score_thresh ** h) * step
                    locations = list(nodes_above_thresh_not_in_clusters)
                    tfce_values[hemi][metric][locations] += single_node_tfce

    return tfce_values


def load_per_subject_scores(args, hemis=HEMIS):
    print("loading per-subject scores")

    all_scores = []

    for subject in tqdm(args.subjects):
        for hemi in hemis:
            results_mod_agnostic_file = get_results_file_path(hemi, subject, searchlight_mode_from_args(args))
            scores = pd.read_csv(results_mod_agnostic_file, index_col=0)

            all_scores.append(scores)

    all_scores = pd.concat(all_scores, ignore_index=True)

    return all_scores


def get_edge_lengths_dicts_based_on_coord_dist(resolution, max_dist="max"):
    path = os.path.join(DATA_DIR, "edge_lengths", resolution, f"edge_lengths_{max_dist}.p")
    if not os.path.isfile(path):
        edge_lengths_dicts = dict()
        fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
        surface_infl = {hemi: surface.load_surf_mesh(fsaverage[f"infl_{hemi}"]) for hemi in HEMIS}

        for hemi in HEMIS:
            coords = surface_infl[hemi].coordinates

            edges = np.vstack([surface_infl[hemi].faces[:, [0, 1]],
                               surface_infl[hemi].faces[:, [0, 2]],
                               surface_infl[hemi].faces[:, [1, 2]]])
            edges = np.array([(e0, e1) if e0 < e1 else (e1, e0) for e0, e1 in edges])
            lengths = np.sqrt(np.sum((coords[edges[:, 0]] - coords[edges[:, 1]]) ** 2, axis=1))
            max_dist = lengths.mean() if max_dist == "mean" else lengths.max()
            print(f"{hemi} hemi max dist for connectivity: {max_dist}")

            all_dists = cdist(coords, coords, metric="euclidean")
            connected = [(x1, x2) for x1, x2 in np.argwhere(all_dists < max_dist) if not (x1 == x2) and (x1 < x2)]
            edge_lengths_dicts[hemi] = {e: all_dists[e] for e in connected}

        os.makedirs(os.path.dirname(path), exist_ok=True)
        pickle.dump(edge_lengths_dicts, open(path, 'wb'))
    else:
        edge_lengths_dicts = pickle.load(open(path, 'rb'))
    # cross_hemi_dists = cdist(surface_infl['left'].coordinates, surface_infl['right'].coordinates, metric="euclidean")
    # np.argwhere(cross_hemi_dists < 20)
    return edge_lengths_dicts


# Adapted from https://github.com/mne-tools/mne-python/blob/maint/1.9/mne/stats/parametric.py#L19-L57
def ttest_1samp_no_p(X, sigma=0, method="relative"):
    """Perform one-sample t-test.

    This is a modified version of :func:`scipy.stats.ttest_1samp` that avoids
    a (relatively) time-consuming p-value calculation, and can adjust
    for implausibly small variance values :footcite:`RidgwayEtAl2012`.

    Parameters
    ----------
    X : array
        Array to return t-values for.
    sigma : float
        The variance estimate will be given by ``var + sigma * max(var)`` or
        ``var + sigma``, depending on "method". By default this is 0 (no
        adjustment). See Notes for details.
    method : str
        If 'relative', the minimum variance estimate will be sigma * max(var),
        if 'absolute' the minimum variance estimate will be sigma.

    Returns
    -------
    t : array
        T-values, potentially adjusted using the hat method.

    """
    var = np.var(X, axis=0, ddof=1)
    if sigma > 0:
        limit = sigma * np.max(var) if method == "relative" else sigma
        var += limit
    return np.mean(X, axis=0) / np.sqrt(var / X.shape[0])


def calc_t_value(values, popmean, sigma=0):
    if np.all(values == values[0]):
        # If all values are equal, the t-value is not defined
        t_val = np.nan
    else:
        t_val = ttest_1samp_no_p(values - popmean, sigma=sigma)
    return t_val


def calc_image_t_values(data, popmean, use_tqdm=False, metric=None, sigma=0):
    iterator = tqdm(data.T, desc=f'calculating t-values for {metric}') if use_tqdm else data.T
    return np.array(
        [calc_t_value(x, popmean, sigma) for x in iterator]
    )


def calc_t_values(scores):
    tvals = {hemi: dict() for hemi in HEMIS}
    for hemi in HEMIS:
        for training_split in tqdm(ATTENTION_MOD_SPLITS, desc=f'calculating {hemi} hemi t vals'):
            testing_splits = [split for split in ATTENTION_MOD_SPLITS if split != training_split]
            for testing_split in testing_splits:
                n_vertices = len(scores[scores.hemi == hemi].vertex.unique())
                data = np.zeros((len(args.subjects), n_vertices))
                for i, subj in enumerate(args.subjects):
                    scores_filtered = scores[
                        (scores.hemi == hemi) & (scores.train_split == training_split) & (
                                scores.test_split == testing_split)]
                    data[i] = scores_filtered[(scores_filtered.subject == subj)].value.values

                popmean = 0.5
                metric_name = f'{training_split}-{testing_split}'
                tvals[hemi][metric_name] = calc_image_t_values(data, popmean)

    return tvals


def calc_test_statistics(args):
    t_values_path = os.path.join(permutation_results_dir(args), "t_values.p")
    if not os.path.isfile(t_values_path):
        print(f"Calculating t-values")
        per_subject_scores = load_per_subject_scores(args)
        t_values = calc_t_values(per_subject_scores)
        pickle.dump(t_values, open(t_values_path, 'wb'))
    else:
        t_values = pickle.load(open(t_values_path, 'rb'))

    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values_{args.metric}.p")
    if not os.path.isfile(tfce_values_path):
        print("calculating tfce..")
        edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)
        tfce_values = calc_tfce_values(t_values, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e,
                                       dh=args.tfce_dh, use_tqdm=True)
        pickle.dump(tfce_values, open(tfce_values_path, "wb"))
    else:
        tfce_values = pickle.load(open(tfce_values_path, 'rb'))

    for hemi in HEMIS:
        print(f"mean tfce value ({hemi} hemi): {np.nanmean(tfce_values[hemi][args.metric]):.2f} | ", end="")
        print(f"max tfce value ({hemi} hemi): {np.nanmax(tfce_values[hemi][args.metric]):.2f}")

    null_distribution_tfce_values_file = os.path.join(
        permutation_results_dir(args),
        f"tfce_values_null_distribution_{args.metric}.p"
    )
    null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))
    significance_cutoff, max_test_statistic_distr = calc_significance_cutoff(null_distribution_tfce_values, args.metric,
                                                                             args.p_value_threshold)

    p_values = {hemi: np.repeat(np.nan, tfce_values[hemi][args.metric].shape) for hemi, t_vals in t_values.items()}
    for hemi in HEMIS:
        # print(f"{hemi} hemi largest test statistic values: ", np.sort(tfce_values[hemi][args.metric])[-5:])
        # print(f"{hemi} hemi largest test statistic null distr values: ", max_test_statistic_distr[-5:])
        for vertex in tqdm(np.argwhere(tfce_values[hemi][args.metric] > 0)[:, 0],
                           desc=f"calculating p values for {hemi} hemi"):
            test_stat = tfce_values[hemi][args.metric][vertex]
            value_index = np.searchsorted(max_test_statistic_distr, test_stat)
            if value_index >= len(max_test_statistic_distr):
                p_value = 1 - (len(max_test_statistic_distr) - 1) / (len(max_test_statistic_distr))
            else:
                p_value = 1 - value_index / len(max_test_statistic_distr)
            p_values[hemi][vertex] = p_value

        print(f"smallest p value ({hemi}): {np.min(p_values[hemi][p_values[hemi] > 0]):.5f}")

    p_values_path = os.path.join(permutation_results_dir(args), f"p_values_{args.metric}.p")
    pickle.dump(p_values, open(p_values_path, mode='wb'))
    return significance_cutoff


def calc_t_values_null_distr(args, out_path):
    def calc_permutation_t_values(vertex_range, permutations, proc_id, tmp_file_path, subjects, hemi,
                                  latents_mode=LIMITED_CANDIDATE_LATENTS, standardized_predictions=True):
        os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)

        with h5py.File(tmp_file_path, 'w') as f:
            dsets = dict()
            for metric in T_VAL_METRICS:
                tvals_shape = (len(permutations), vertex_range[1] - vertex_range[0])
                dsets[metric] = f.create_dataset(metric, tvals_shape, dtype='float16')

            preloaded_scores = dict()
            for subj in subjects:
                preloaded_scores[subj] = dict()
                for training_mode in TRAINING_MODES:
                    preloaded_scores[subj][training_mode] = dict()

                    if training_mode == MODALITY_AGNOSTIC:
                        feats_config = LatentFeatsConfig(
                            args.model,
                            args.features,
                            args.test_features,
                            args.vision_features,
                            args.lang_features,
                            logging=False
                        )
                    elif training_mode == MODALITY_SPECIFIC_IMAGES:
                        feats_config = LatentFeatsConfig(
                            args.mod_specific_images_model,
                            args.mod_specific_images_features,
                            args.mod_specific_images_test_features,
                            args.vision_features,
                            args.lang_features,
                            logging=False
                        )
                    elif training_mode == MODALITY_SPECIFIC_CAPTIONS:
                        feats_config = LatentFeatsConfig(
                            args.mod_specific_captions_model,
                            args.mod_specific_captions_features,
                            args.mod_specific_captions_test_features,
                            args.vision_features,
                            args.lang_features,
                            logging=False
                        )
                    else:
                        raise RuntimeError(f"Unknown training mode: {training_mode}")

                    base_path = get_results_file_path(
                        feats_config, hemi, subj, training_mode,
                        searchlight_mode_from_args(args), args.l2_regularization_alpha,
                    )
                    if proc_id == args.n_jobs - 1:
                        vertex_iter = trange(vertex_range[0], vertex_range[1],
                                             desc=f"loading scores for {subj} {hemi} hemi {training_mode} decoder")
                    else:
                        vertex_iter = range(vertex_range[0], vertex_range[1])
                    preloaded_scores[subj][training_mode] = []

                    gathered_over_vertices = dict()

                    for vertex_id in vertex_iter:
                        gathered_over_vertices[vertex_id] = []
                        scores_path = os.path.join(os.path.dirname(base_path), "null_distr",
                                                   f"{vertex_id:010d}.p")
                        scores_vertex = pickle.load(open(scores_path, "rb"))
                        for scores_perm in scores_vertex:
                            scores_perm = scores_perm[(scores_perm.latents == latents_mode) & (
                                    scores_perm.standardized_predictions == standardized_predictions)]
                            metrics = scores_perm[['metric', 'value']].set_index('metric').value.to_dict()
                            gathered_over_vertices[vertex_id].append(metrics)
                    for perm_id in range(len(gathered_over_vertices[vertex_range[0]])):
                        gathered = {metric: np.array([gathered_over_vertices[i][perm_id][metric] for i in
                                                      range(vertex_range[0], vertex_range[1])]) for metric in
                                    gathered_over_vertices[vertex_range[0]][perm_id].keys()}
                        # saving in format [subj][training_mode][perm_id][metric]
                        preloaded_scores[subj][training_mode].append(gathered)

            if proc_id == args.n_jobs - 1:
                permutations_iterator = tqdm(enumerate(permutations), total=len(permutations),
                                             desc="calculating null distr t-vals")
            else:
                permutations_iterator = enumerate(permutations)

            for iteration, permutation in permutations_iterator:
                t_values = dict()
                for metric in T_VAL_METRICS:
                    data = np.zeros((len(args.subjects), vertex_range[1] - vertex_range[0]))
                    for i, (idx, subj) in enumerate(zip(permutation, args.subjects)):
                        if metric.split('$')[0] == DIFF:
                            training_mode, metric_name_1, metric_name_2 = metric.split('$')[1:]
                            data[i] = preloaded_scores[subj][training_mode][idx][metric_name_1] - \
                                      preloaded_scores[subj][training_mode][idx][metric_name_2]
                        elif metric.split('$')[0] == DIFF_DECODERS:
                            training_mode_1, training_mode_2, metric_name = metric.split('$')[1:]
                            data[i] = preloaded_scores[subj][training_mode_1][idx][metric_name] - \
                                      preloaded_scores[subj][training_mode_2][idx][metric_name]
                        else:
                            training_mode, metric_name = metric.split('$')
                            data[i] = preloaded_scores[subj][training_mode][idx][metric_name]

                    popmean = 0 if metric.split('$')[0] in [DIFF, DIFF_DECODERS] else 0.5
                    t_values[metric] = calc_image_t_values(data, popmean)
                    dsets[metric][iteration] = t_values[metric].astype(np.float16)

    feats_config = LatentFeatsConfig(
        args.model,
        args.features,
        args.test_features,
        args.vision_features,
        args.lang_features,
        logging=False
    )
    base_path = get_results_file_path(
        feats_config, HEMIS[0], args.subjects[0], MODALITY_AGNOSTIC,
        searchlight_mode_from_args(args), args.l2_regularization_alpha,
    )
    scores_dir = os.path.join(os.path.dirname(base_path), "null_distr")
    null_distr_filepaths = list(glob(os.path.join(scores_dir, "*.p")))
    n_vertices = len(null_distr_filepaths)

    n_permutations = len(pickle.load(open(null_distr_filepaths[0], "rb")))
    permutations_iter = itertools.permutations(range(n_permutations), len(args.subjects))
    permutations = [next(permutations_iter) for _ in range(args.n_permutations_group_level)]

    print('n_permutations: ', n_permutations)
    print('n_vertices: ', n_vertices)

    n_per_job = math.ceil(n_vertices / args.n_jobs)
    print(f"n vertices per job: {n_per_job}")

    vertex_ranges = [(job_id * n_per_job, min((job_id + 1) * n_per_job, n_vertices)) for job_id in range(args.n_jobs)]
    print('vertex ranges for jobs: ', vertex_ranges)

    tmp_filenames = dict()
    for hemi in HEMIS:
        tmp_filenames[hemi] = {job_id: os.path.join(os.path.dirname(out_path), f"temp_t_vals", f"{job_id}_{hemi}.hdf5")
                               for job_id in range(args.n_jobs)}

        # TODO single iter for debugging
        # calc_permutation_t_values(vertex_ranges[-1], permutations, args.n_jobs - 1, tmp_filenames[hemi][args.n_jobs - 1],
        #                           args.subjects, hemi)
        Parallel(n_jobs=args.n_jobs, mmap_mode=None, max_nbytes=None)(
            delayed(calc_permutation_t_values)(
                vertex_ranges[id],
                permutations,
                id,
                tmp_filenames[hemi][id],
                args.subjects,
                hemi,
            )
            for id in range(args.n_jobs)
        )
        print(f'finished calculating null distr t-vals for {hemi} hemi')

    with h5py.File(out_path, 'w') as all_t_vals_file:
        for hemi in HEMIS:
            tmp_files = {job_id: h5py.File(tmp_filenames[hemi][job_id], 'r') for job_id in range(args.n_jobs)}

            for metric in tmp_files[0].keys():
                tvals_shape = (args.n_permutations_group_level, n_vertices)
                all_t_vals_file.create_dataset(f'{hemi}__{metric}', tvals_shape, dtype='float32', fillvalue=np.nan)

            for i in tqdm(range(args.n_permutations_group_level), desc="assembling results"):
                for metric in tmp_files[0].keys():
                    data_tvals = np.concatenate([tmp_files[job_id][metric][i] for job_id in range(args.n_jobs)])
                    all_t_vals_file[f'{hemi}__{metric}'][i] = data_tvals

    print("finished assemble")


def permutation_results_dir(args):
    return str(os.path.join(
        SEARCHLIGHT_CLASSIFICATION_PERMUTATION_TESTING_RESULTS_DIR,
        args.resolution, searchlight_mode_from_args(args)
    ))


def create_null_distribution(args):
    tfce_values_null_distribution_path = os.path.join(
        permutation_results_dir(args), f"tfce_values_null_distribution_{args.metric}.p"
    )
    if not os.path.isfile(tfce_values_null_distribution_path):
        t_values_null_distribution_path = os.path.join(
            permutation_results_dir(args), f"t_values_null_distribution.hdf5"
        )
        if not os.path.isfile(t_values_null_distribution_path):
            print(f"Calculating t-values: null distribution")
            os.makedirs(os.path.dirname(t_values_null_distribution_path), exist_ok=True)
            calc_t_values_null_distr(args, t_values_null_distribution_path)

        edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)

        def tfce_values_job(n_per_job, edge_lengths, proc_id, t_vals_null_distr_path):
            with h5py.File(t_vals_null_distr_path, 'r') as t_vals:
                indices = range(proc_id * n_per_job, min((proc_id + 1) * n_per_job, args.n_permutations_group_level))
                iterator = tqdm(indices,
                                desc="Calculating tfce values for null distribution") if proc_id == args.n_jobs - 1 else indices
                tfce_values = []
                for iteration in iterator:
                    vals = {hemi: {metric: t_vals[f"{hemi}__{metric}"][iteration] for metric in T_VAL_METRICS} for hemi
                            in HEMIS}
                    tfce_values.append(
                        calc_tfce_values(
                            vals, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e, dh=args.tfce_dh,
                        )
                    )
                return tfce_values

        n_per_job = math.ceil(args.n_permutations_group_level / args.n_jobs)
        # TODO for debugging
        # tfce_values_job(n_per_job, edge_lengths.copy(), args.n_jobs-1, t_values_null_distribution_path)
        tfce_values = Parallel(n_jobs=args.n_jobs)(
            delayed(tfce_values_job)(
                n_per_job,
                edge_lengths.copy(),
                id,
                t_values_null_distribution_path,
            )
            for id in range(args.n_jobs)
        )
        tfce_values = np.concatenate(tfce_values)

        pickle.dump(tfce_values, open(tfce_values_null_distribution_path, 'wb'))


def add_searchlight_permutation_args(parser):
    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS_ADDITIONAL_TEST)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)
    parser.add_argument("--tfce-dh", type=float, default=0.1)

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-permutations-group-level", type=int, default=1000)

    parser.add_argument("--p-value-threshold", type=float, default=DEFAULT_P_VAL_THRESHOLD)
    parser.add_argument("--tfce-value-threshold", type=float, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # create_null_distribution(args)

    for training_split in ATTENTION_MOD_SPLITS:
        testing_splits = [split for split in ATTENTION_MOD_SPLITS if split != training_split]
        for testing_split in testing_splits:
            print(f"\n\nPermutation Testing for train: {training_split} | test: {testing_split}\n")
            metric_name = f'{training_split}-{testing_split}'
            args.metric = metric_name
            significance_cutoff = calc_test_statistics(args)
            # create_masks(permutation_results_dir(args), args.metric, significance_cutoff, args.tfce_value_threshold,
            #              args.resolution, args.radius, args.n_neighbors)
