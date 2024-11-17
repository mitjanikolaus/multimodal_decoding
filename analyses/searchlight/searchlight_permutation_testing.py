import argparse
import copy
import hashlib
import itertools
import math
import warnings

import h5py
import numpy as np
from joblib import Parallel, delayed
from nilearn import datasets
import os
from glob import glob
import pickle

from nilearn.surface import surface
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from tqdm import tqdm

from analyses.ridge_regression_decoding import MOD_SPECIFIC_CAPTIONS, MOD_SPECIFIC_IMAGES, MODE_AGNOSTIC, ACC_CAPTIONS, \
    ACC_IMAGES
from analyses.searchlight.searchlight import SEARCHLIGHT_OUT_DIR, \
    METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, \
    SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, METRIC_CROSS_DECODING
from utils import SUBJECTS, HEMIS, DEFAULT_RESOLUTION, FS_HEMI_NAMES, export_to_gifti, ACC_IMAGERY_WHOLE_TEST, \
    ACC_IMAGERY, ACC_CROSS_CAPTIONS_TO_IMAGES, ACC_CROSS_IMAGES_TO_CAPTIONS

DEFAULT_N_JOBS = 10

CHANCE_VALUES = {
    ACC_CAPTIONS: 0.5,
    ACC_IMAGES: 0.5,
    METRIC_DIFF_IMAGES: 0,
    METRIC_DIFF_CAPTIONS: 0,
    METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC: 0,
    ACC_IMAGERY: 0.5,
    ACC_IMAGERY_WHOLE_TEST: 0.5,
}


def process_scores(scores_agnostic, scores_mod_specific_captions, scores_mod_specific_images, nan_locations):
    scores = dict()

    for metric in [ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST]:
        scores[metric] = np.repeat(np.nan, nan_locations.shape)
        scores[metric][~nan_locations] = np.array([score[metric] for score in scores_agnostic])

    if scores_mod_specific_captions is not None and scores_mod_specific_images is not None:
        scores_specific_captions = dict()
        for metric in [ACC_CAPTIONS, ACC_IMAGES, ACC_CROSS_CAPTIONS_TO_IMAGES]:
            scores_specific_captions[metric] = np.repeat(np.nan, nan_locations.shape)
            scores_specific_captions[metric][~nan_locations] = np.array(
                [score[metric] for score in scores_mod_specific_captions])

        scores_specific_images = dict()
        for metric in [ACC_CAPTIONS, ACC_IMAGES, ACC_CROSS_IMAGES_TO_CAPTIONS]:
            scores_specific_images[metric] = np.repeat(np.nan, nan_locations.shape)
            scores_specific_images[metric][~nan_locations] = np.array(
                [score[metric] for score in scores_mod_specific_images])

        scores[METRIC_DIFF_IMAGES] = np.array(
            [ai - si for ai, ac, si, sc in
             zip(scores[ACC_IMAGES],
                 scores[ACC_CAPTIONS],
                 scores_specific_images[ACC_IMAGES],
                 scores_specific_captions[ACC_CAPTIONS])]
        )
        scores[METRIC_DIFF_CAPTIONS] = np.array(
            [ac - sc for ai, ac, si, sc in
             zip(scores[ACC_IMAGES],
                 scores[ACC_CAPTIONS],
                 scores_specific_images[ACC_IMAGES],
                 scores_specific_captions[ACC_CAPTIONS])]
        )

        scores[ACC_CROSS_IMAGES_TO_CAPTIONS] = scores_specific_images[ACC_CROSS_IMAGES_TO_CAPTIONS]
        scores[ACC_CROSS_CAPTIONS_TO_IMAGES] = scores_specific_captions[ACC_CROSS_CAPTIONS_TO_IMAGES]

    return scores


def load_per_subject_scores(args, return_nan_locations_and_n_neighbors=False):
    print("loading per-subject scores")

    per_subject_scores = {subj: dict() for subj in args.subjects}
    per_subject_n_neighbors = {subj: dict() for subj in args.subjects}
    per_subject_nan_locations = {subj: dict() for subj in args.subjects}

    for subject in tqdm(args.subjects):
        for hemi in HEMIS:
            results_agnostic_file = os.path.join(
                SEARCHLIGHT_OUT_DIR,
                f'{MODE_AGNOSTIC}/{args.model}/{args.features}/{subject}/{args.resolution}/{hemi}/{args.mode}/'
                f'alpha_{str(args.l2_regularization_alpha)}.p'
            )
            results_agnostic = pickle.load(open(results_agnostic_file, 'rb'))
            scores_agnostic = results_agnostic['scores']
            nan_locations = results_agnostic['nan_locations']
            n_neighbors = results_agnostic['n_neighbors'] if 'n_neighbors' in results_agnostic else None
            per_subject_n_neighbors[subject][hemi] = n_neighbors
            per_subject_nan_locations[subject][hemi] = nan_locations

            results_mod_specific_vision_file = os.path.join(
                SEARCHLIGHT_OUT_DIR,
                f'{MOD_SPECIFIC_IMAGES}/{args.mod_specific_vision_model}/{args.mod_specific_vision_features}/{subject}/'
                f'{args.resolution}/{hemi}/{args.mode}/alpha_{str(args.l2_regularization_alpha)}.p'
            )
            if os.path.isfile(results_mod_specific_vision_file):
                scores_images = pickle.load(open(results_mod_specific_vision_file, 'rb'))['scores']
            else:
                print(f"Missing modality-specific results: {results_mod_specific_vision_file}")
                scores_images = None

            results_mod_specific_lang_file = os.path.join(
                SEARCHLIGHT_OUT_DIR,
                f'{MOD_SPECIFIC_CAPTIONS}/{args.mod_specific_lang_model}/{args.mod_specific_lang_features}/{subject}/'
                f'{args.resolution}/{hemi}/{args.mode}/alpha_{str(args.l2_regularization_alpha)}.p'
            )
            if os.path.isfile(results_mod_specific_lang_file):
                scores_captions = pickle.load(open(results_mod_specific_lang_file, 'rb'))['scores']
            else:
                print(f"Missing modality-specific results: {results_mod_specific_lang_file}")
                scores_captions = None

            scores = process_scores(scores_agnostic, scores_captions, scores_images, nan_locations)

            # print({n: round(np.nanmean(score), 4) for n, score in scores.items()})
            # print({f"{n}_max": round(np.nanmax(score), 2) for n, score in scores.items()})
            # print("")

            per_subject_scores[subject][hemi] = scores

    if return_nan_locations_and_n_neighbors:
        return per_subject_scores, per_subject_nan_locations, per_subject_n_neighbors
    else:
        return per_subject_scores


def compute_adjacency_matrix(surface, values='ones'):
    """Computes the adjacency matrix for a surface.
    The adjacency matrix is a matrix with one row and one column for each vertex
    such that the value of a cell `(u,v)` in the matrix is 1 if nodes `u` and
    `v` are adjacent and 0 otherwise.
    Parameters
    ----------
    surface : Surface-like
        The surface whose adjacency matrix is to be computed.
    values : { 'euclidean' | 'inveuclidean' | 'ones'}, optional
        If `values` is `'ones'` (the default), then the returned matrix
        contains uniform values in the cells representing edges. If the value is
        `'euclidean'` then the cells contain the edge length of the represented
        edge. If the value is `'inveuclidean'`, then the inverse of the distances
        are returned.
    dtype : numpy dtype-like or None, optional
        The dtype that should be used for the returned sparse matrix.
    Returns
    -------
    matrix : scipy.sparse.csr_matrix
        A sparse matrix representing the edge relationships in `surface`.
    """
    n = surface.coordinates.shape[0]
    edges = np.vstack([surface.faces[:, [0, 1]],
                       surface.faces[:, [0, 2]],
                       surface.faces[:, [1, 2]]])

    bigcol = edges[:, 0] > edges[:, 1]
    lilcol = ~bigcol
    edges = np.concatenate([edges[bigcol, 0] + edges[bigcol, 1] * n,
                            edges[lilcol, 1] + edges[lilcol, 0] * n])
    edges = np.unique(edges)
    (u, v) = (edges // n, edges % n)
    # Calculate distances between pairs. We use this as a weighting to make sure that
    # smoothing takes into account the distance between each vertex neighbor
    if values == 'euclidean' or values == 'inveuclidean':
        coords = surface.coordinates
        edge_lens = np.sqrt(np.sum((coords[u, :] - coords[v, :]) ** 2, axis=1))
        if values == 'inveuclidean':
            edge_lens = 1 / edge_lens
    elif values == 'ones':
        edge_lens = np.ones(edges.shape)
    else:
        raise ValueError(f"unrecognized values argument: {values}")
    # We can now make a sparse matrix.
    ee = np.concatenate([edge_lens, edge_lens])
    uv = np.concatenate([u, v])
    vu = np.concatenate([v, u])
    return csr_matrix((ee, (uv, vu)), shape=(n, n))


def _compute_vertex_neighborhoods(surface):
    """For each vertex, compute the neighborhood.
    The neighborhood is  defined as all the vertices that are connected by a
    face.

    Parameters
    ----------
    surface : Surface-like
        The surface whose vertex neighborhoods are to be computed.

    Returns
    -------
    neighbors : list
        A list of all the vertices that are connected to each vertex
    """
    from scipy.sparse import find
    matrix = compute_adjacency_matrix(surface)
    return [find(row)[1] for row in matrix]


def get_edge_lengths_dicts_based_on_coord_dist(resolution, max_dist="max"):
    path = os.path.join(SEARCHLIGHT_OUT_DIR, "edge_lengths", resolution, f"edge_lengths_{max_dist}.p")
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


def calc_tfce_values(t_values, edge_lengths_dicts, metric, h=2, e=1, dh=0.1, cluster_extents_measure="num_vertices"):
    tfce_values = dict()

    for hemi in HEMIS:
        values = t_values[hemi][metric]
        max_score = np.nanmax(values)
        if np.isnan(max_score) or np.isinf(max_score):
            print("encountered NaN or Inf in t-values while calculating tfce values")
            tfce_values[hemi] = {metric: np.zeros_like(values)}
            continue

        if max_score <= 0:
            tfce_values[hemi] = {metric: np.zeros_like(values)}
            continue

        step = max_score / 100 if dh == "auto" else dh
        score_threshs = np.arange(step, max_score + step, step)

        tfce_values[hemi] = {metric: np.zeros_like(values)}

        for score_thresh in score_threshs:
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


def calc_t_value(values, popmean, epsilon=1e-8):
    # use heuristic (mean needs to be greater than popmean) to speed up calculation
    values_no_nan = values[~np.isnan(values)]
    if values_no_nan.mean() > popmean:
        t_val = stats.ttest_1samp(values_no_nan, popmean=popmean, alternative="greater")[0]
        if np.isinf(t_val):
            # Add/subtract epsilon for numerical stability
            values_no_nan[0] = values_no_nan[0] + epsilon
            values_no_nan[-1] = values_no_nan[-1] - epsilon
            t_val = stats.ttest_1samp(values_no_nan, popmean=popmean, alternative="greater")[0]
            print(f"corrected t val: {t_val}")
        return t_val
    else:
        return 0


def calc_image_t_values(data, popmean, use_tqdm=False, t_vals_cache=None, precision=2, epsilon=1e-8):
    data = data.round(precision)
    iterator = tqdm(data.T) if use_tqdm else data.T
    if t_vals_cache is None:
        return np.array(
            [calc_t_value(x, popmean, epsilon) for x in iterator]
        )
    else:
        t_vals = []
        for x in iterator:
            x_no_nan = x[~np.isnan(x)]
            if x_no_nan.mean() > popmean:
                key = hashlib.sha1(np.sort(x_no_nan)).hexdigest()
                if key in t_vals_cache:
                    t_vals.append(t_vals_cache[key])
                else:
                    t_val = stats.ttest_1samp(x_no_nan, popmean=popmean, alternative="greater")[0]
                    if np.isinf(t_val):
                        # Add/subtract epsilon for numerical stability
                        x_no_nan[0] = x_no_nan[0] + epsilon
                        x_no_nan[-1] = x_no_nan[-1] - epsilon
                        t_val = stats.ttest_1samp(x_no_nan, popmean=popmean, alternative="greater")[0]
                    t_vals.append(t_val)
                    t_vals_cache[key] = t_val
            else:
                # mean is below popmean, t value won't be significant
                t_vals.append(0)

        return np.array(t_vals)


def calc_t_values(per_subject_scores):
    t_values = {hemi: dict() for hemi in HEMIS}
    for hemi in HEMIS:
        for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, ACC_IMAGES, ACC_CAPTIONS,
                       ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, ACC_CROSS_IMAGES_TO_CAPTIONS, ACC_CROSS_CAPTIONS_TO_IMAGES]:
            data = np.array([per_subject_scores[subj][hemi][metric] for subj in args.subjects])
            popmean = CHANCE_VALUES[metric]
            enough_data = np.argwhere(((~np.isnan(data)).sum(axis=0)) > 2)[:, 0]  # at least 3 datapoints
            t_values[hemi][metric] = np.repeat(np.nan, data.shape[1])
            t_values[hemi][metric][enough_data] = calc_image_t_values(data[:, enough_data], popmean, True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_values[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC] = np.nanmin(
                (
                    t_values[hemi][METRIC_DIFF_CAPTIONS],
                    t_values[hemi][METRIC_DIFF_IMAGES],
                    t_values[hemi][ACC_IMAGES],
                    t_values[hemi][ACC_CAPTIONS]),
                axis=0)
            t_values[hemi][METRIC_CROSS_DECODING] = np.nanmin(
                (t_values[hemi][ACC_CROSS_IMAGES_TO_CAPTIONS],
                 t_values[hemi][ACC_CROSS_CAPTIONS_TO_IMAGES]),
                axis=0
            )

    return t_values


def calc_significance_cutoff(args, p_value_threshold=0.05):
    null_distribution_tfce_values_file = os.path.join(
        permutation_results_dir(args),
        f"tfce_values_null_distribution{get_hparam_suffix(args)}.p"
    )
    print("loading null distribution test statistic: ", null_distribution_tfce_values_file)
    null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))

    max_test_statistic_distr = sorted([
        np.nanmax(np.concatenate((n[HEMIS[0]][args.metric], n[HEMIS[1]][args.metric])))
        for n in null_distribution_tfce_values
    ])

    significance_cutoff = np.quantile(max_test_statistic_distr, 1 - p_value_threshold)
    print(f"{len(null_distribution_tfce_values)} permutations")
    print(f"cluster test statistic significance cutoff for p<{p_value_threshold} (across hemis): {significance_cutoff}")

    for thresh in [0.05, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        val = np.quantile(max_test_statistic_distr, 1 - thresh)
        print(f"(info) cluster test statistic significance cutoff for p<{thresh} (across hemis): {val}")

    return significance_cutoff, max_test_statistic_distr


def calc_test_statistics(args):
    t_values_path = os.path.join(permutation_results_dir(args), "t_values.p")
    if not os.path.isfile(t_values_path):
        os.makedirs(os.path.dirname(t_values_path), exist_ok=True)
        per_subject_scores = load_per_subject_scores(args)
        print(f"Calculating t-values")
        t_values = calc_t_values(per_subject_scores)
        pickle.dump(t_values, open(t_values_path, 'wb'))
    else:
        t_values = pickle.load(open(t_values_path, 'rb'))

    print("calculating tfce..")
    edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)
    tfce_values = calc_tfce_values(t_values, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e)

    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
    pickle.dump(tfce_values, open(tfce_values_path, "wb"))

    for hemi in HEMIS:
        print(f"mean tfce value ({hemi} hemi): {np.nanmean(tfce_values[hemi][args.metric]):.2f} | ", end="")
        print(f"max tfce value ({hemi} hemi): {np.nanmax(tfce_values[hemi][args.metric]):.2f}")

    significance_cutoff, max_test_statistic_distr = calc_significance_cutoff(args)

    p_values = {hemi: np.repeat(np.nan, t_values[hemi][args.metric].shape) for hemi, t_vals in t_values.items()}
    for hemi in HEMIS:
        print(f"{hemi} hemi largest test statistic values: ",
              sorted([t for t in tfce_values[hemi][args.metric]], reverse=True)[:10])
        print("calculating p values..")
        for vertex in tqdm(np.argwhere(tfce_values[hemi][args.metric] > 0)[:, 0]):
            test_stat = tfce_values[hemi][args.metric][vertex]
            value_index = np.searchsorted(max_test_statistic_distr, test_stat)
            if value_index >= len(max_test_statistic_distr):
                p_value = 1 - (len(max_test_statistic_distr) - 1) / (len(max_test_statistic_distr))
            else:
                p_value = 1 - value_index / len(max_test_statistic_distr)
            p_values[hemi][vertex] = p_value

        print(f"smallest p value ({hemi}): {np.min(p_values[hemi][p_values[hemi] > 0]):.5f}")

    p_values_path = os.path.join(permutation_results_dir(args), f"p_values{get_hparam_suffix(args)}.p")
    pickle.dump(p_values, open(p_values_path, mode='wb'))


def load_null_distr_per_subject_scores(args):
    per_subject_scores_null_distr = []

    for subject in tqdm(args.subjects):
        for hemi in HEMIS:
            results_agnostic_file = os.path.join(
                SEARCHLIGHT_OUT_DIR,
                f'{MODE_AGNOSTIC}/{args.model}/{args.features}/{subject}/{args.resolution}/{hemi}/{args.mode}/'
                f'alpha_{str(args.l2_regularization_alpha)}.p'
            )
            results_agnostic = pickle.load(open(results_agnostic_file, 'rb'))
            nan_locations = results_agnostic['nan_locations']

            results_mod_specific_vision_file = os.path.join(
                SEARCHLIGHT_OUT_DIR,
                f'{MOD_SPECIFIC_IMAGES}/{args.mod_specific_vision_model}/{args.mod_specific_vision_features}/{subject}/'
                f'{args.resolution}/{hemi}/{args.mode}/alpha_{str(args.l2_regularization_alpha)}.p'
            )
            results_mod_specific_lang_file = os.path.join(
                SEARCHLIGHT_OUT_DIR,
                f'{MOD_SPECIFIC_CAPTIONS}/{args.mod_specific_lang_model}/{args.mod_specific_lang_features}/{subject}/'
                f'{args.resolution}/{hemi}/{args.mode}/alpha_{str(args.l2_regularization_alpha)}.p'
            )

            def load_null_distr_scores(base_path):
                scores_dir = os.path.join(base_path, "null_distr")
                print(f'loading scores from {scores_dir}')
                score_paths = sorted(list(glob(os.path.join(scores_dir, "*.p"))))
                if len(score_paths) == 0:
                    raise RuntimeError(f"No null distribution scores found: {scores_dir}")
                last_idx = int(os.path.basename(score_paths[-1])[:-2])
                assert last_idx == len(score_paths) - 1, f"{last_idx} vs. {len(score_paths)}"

                def load_scores_from_pickle(paths, proc_id):
                    job_scores = []
                    iterator = tqdm(paths) if proc_id == 0 else paths
                    for path in iterator:
                        scores = pickle.load(open(path, "rb"))
                        job_scores.append(scores)
                    return job_scores

                n_per_job = math.ceil(len(score_paths) / args.n_jobs)
                all_scores = Parallel(n_jobs=args.n_jobs)(
                    delayed(load_scores_from_pickle)(
                        score_paths[id * n_per_job:(id + 1) * n_per_job],
                        id,
                    )
                    for id in range(args.n_jobs)
                )
                return np.concatenate(all_scores)

            null_distribution_agnostic = load_null_distr_scores(os.path.dirname(results_agnostic_file))
            null_distribution_images = load_null_distr_scores(os.path.dirname(results_mod_specific_vision_file))
            null_distribution_captions = load_null_distr_scores(os.path.dirname(results_mod_specific_lang_file))

            num_permutations = len(null_distribution_agnostic[0])
            print('final per subject scores null distribution dict creation:')
            for i in tqdm(range(num_permutations)):
                distr = [null_distr[i] for null_distr in null_distribution_agnostic]
                distr_caps = [null_distr[i] for null_distr in null_distribution_captions]
                distr_imgs = [null_distr[i] for null_distr in null_distribution_images]
                if len(per_subject_scores_null_distr) <= i:
                    per_subject_scores_null_distr.append({subj: dict() for subj in args.subjects})
                scores = process_scores(distr, distr_caps, distr_imgs, nan_locations)
                per_subject_scores_null_distr[i][subject][hemi] = scores
    return per_subject_scores_null_distr


def calc_t_values_null_distr(args, out_path):
    per_subject_scores_null_distr_path = os.path.join(
        permutation_results_dir(args), f"per_subject_scores_null_distr.p"
    )
    if not os.path.isfile(per_subject_scores_null_distr_path):
        print("loading per subject null distr scores")
        per_subject_scores_null_distr = load_null_distr_per_subject_scores(args)
        os.makedirs(os.path.dirname(per_subject_scores_null_distr_path), exist_ok=True)
        pickle.dump(per_subject_scores_null_distr, open(per_subject_scores_null_distr_path, 'wb'))
    else:
        per_subject_scores_null_distr = pickle.load(open(per_subject_scores_null_distr_path, 'rb'))

    def calc_permutation_t_values(per_subject_scores, permutations, proc_id, tmp_file_path, subjects):
        os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)

        with h5py.File(tmp_file_path, 'w') as f:
            dsets = dict()
            for hemi in HEMIS:
                dsets[hemi] = dict()
                for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, ACC_IMAGES, ACC_CAPTIONS,
                               METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST]:
                    tvals_shape = (len(permutations), per_subject_scores[0][subjects[0]][hemi][ACC_IMAGES].size)
                    dsets[hemi][metric] = f.create_dataset(f"{hemi}__{metric}", tvals_shape, dtype='float16')

            iterator = tqdm(enumerate(permutations), total=len(permutations)) if proc_id == 0 else enumerate(
                permutations)
            t_vals_cache = {}
            for iteration, permutation in iterator:
                t_values = {hemi: dict() for hemi in HEMIS}
                for hemi in HEMIS:
                    for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, ACC_IMAGES, ACC_CAPTIONS,
                                   ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, ACC_CROSS_IMAGES_TO_CAPTIONS,
                                   ACC_CROSS_CAPTIONS_TO_IMAGES]:
                        data = np.array(
                            [per_subject_scores[idx][subj][hemi][metric] for idx, subj in
                             zip(permutation, args.subjects)])
                        popmean = CHANCE_VALUES[metric]
                        t_values[hemi][metric] = calc_image_t_values(data, popmean, t_vals_cache=t_vals_cache)
                        dsets[hemi][metric][iteration] = t_values[hemi][metric]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        dsets[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC][iteration] = np.nanmin(
                            (
                                t_values[hemi][METRIC_DIFF_CAPTIONS],
                                t_values[hemi][METRIC_DIFF_IMAGES],
                                t_values[hemi][ACC_IMAGES],
                                t_values[hemi][ACC_CAPTIONS]),
                            axis=0)
                        dsets[hemi][METRIC_CROSS_DECODING][iteration] = np.nanmin(
                            (t_values[hemi][ACC_CROSS_IMAGES_TO_CAPTIONS],
                             t_values[hemi][ACC_CROSS_CAPTIONS_TO_IMAGES]),
                            axis=0
                        )

    permutations_iter = itertools.permutations(range(len(per_subject_scores_null_distr)), len(args.subjects))
    permutations = [next(permutations_iter) for _ in range(args.n_permutations_group_level)]

    n_vertices = per_subject_scores_null_distr[0][args.subjects[0]][HEMIS[0]][ACC_IMAGES].shape[0]
    enough_data = {
        hemi: np.argwhere(
            (~np.isnan([per_subject_scores_null_distr[0][subj][hemi][ACC_IMAGES] for subj in args.subjects])).sum(
                axis=0) > 2)[:, 0]
        for hemi in HEMIS
    }  # at least 3 datapoints
    enough_data_lengths = {hemi: len(e) for hemi, e in enough_data.items()}
    print(f"original n vertices: {n_vertices} | enough data: {enough_data_lengths}")

    n_per_job = {hemi: math.ceil(len(enough_data[hemi]) / args.n_jobs) for hemi in HEMIS}
    print(f"n vertices per job: {n_per_job}")

    print("filtering scores for enough data and splitting up for jobs")
    scores_jobs = {job_id: [] for job_id in range(args.n_jobs)}
    for id, scores in tqdm(enumerate(per_subject_scores_null_distr), total=len(per_subject_scores_null_distr)):
        for job_id in range(args.n_jobs):
            scores_jobs[job_id].append({s: {hemi: dict() for hemi in HEMIS} for s in args.subjects})
        for subj in args.subjects:
            for hemi in HEMIS:
                for metric in scores[subj][hemi].keys():
                    for job_id in range(args.n_jobs):
                        filtered = scores[subj][hemi][metric][enough_data[hemi]]
                        scores_jobs[job_id][id][subj][hemi][metric] = filtered[
                                                                      job_id * n_per_job[hemi]:(job_id + 1) * n_per_job[
                                                                          hemi]]

    tmp_filenames = {job_id: os.path.join(os.path.dirname(out_path), "temp_t_vals", f"{job_id}.hdf5") for job_id in
                     range(args.n_jobs)}
    Parallel(n_jobs=args.n_jobs, mmap_mode=None, max_nbytes=None)(
        delayed(calc_permutation_t_values)(
            scores_jobs[id],
            permutations,
            id,
            tmp_filenames[id],
            args.subjects,
        )
        for id in range(args.n_jobs)
    )

    print("assembling results")
    tmp_files = dict()
    for job_id in range(args.n_jobs):
        tmp_files[job_id] = h5py.File(tmp_filenames[job_id], 'r')

    with h5py.File(out_path, 'w') as all_t_vals_file:
        for hemi_metric in tmp_files[0].keys():
            tvals_shape = (args.n_permutations_group_level, n_vertices)
            all_t_vals_file.create_dataset(hemi_metric, tvals_shape, dtype='float16', fillvalue=np.nan)

        for i in tqdm(range(args.n_permutations_group_level)):
            for hemi_metric in tmp_files[0].keys():
                hemi = hemi_metric.split('__')[0]
                data_tvals = np.repeat(np.nan, n_vertices)
                data_tvals[enough_data[hemi]] = np.concatenate(
                    [tmp_files[job_id][hemi_metric][i] for job_id in range(args.n_jobs)])
                all_t_vals_file[hemi_metric][i] = data_tvals

    print("finished assemble")


def permutation_results_dir(args):
    return str(os.path.join(
        SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, args.model, args.features, args.mod_specific_vision_model,
        args.mod_specific_vision_features, args.mod_specific_lang_model, args.mod_specific_lang_features,
        args.resolution, args.mode
    ))


def get_hparam_suffix(args):
    return f"_{args.metric}_h_{args.tfce_h}_e_{args.tfce_e}"


def create_null_distribution(args):
    tfce_values_null_distribution_path = os.path.join(
        permutation_results_dir(args), f"tfce_values_null_distribution{get_hparam_suffix(args)}.p"
    )
    if not os.path.isfile(tfce_values_null_distribution_path):
        t_values_null_distribution_path = os.path.join(
            permutation_results_dir(args), f"t_values_null_distribution.hdf5"
        )
        if not os.path.isfile(t_values_null_distribution_path):
            print(f"Calculating t-values: null distribution")
            os.makedirs(os.path.dirname(t_values_null_distribution_path), exist_ok=True)
            calc_t_values_null_distr(args, t_values_null_distribution_path)

        print(f"Calculating tfce values for null distribution")
        edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)

        def tfce_values_job(n_per_job, edge_lengths, proc_id, t_vals_null_distr_path):
            with h5py.File(t_vals_null_distr_path, 'r') as t_vals:
                # t_values = [id * n_per_job: (id + 1) * n_per_job]
                indices = range(proc_id * n_per_job, min((proc_id + 1) * n_per_job, args.n_permutations_group_level))
                iterator = tqdm(indices) if proc_id == 0 else indices
                tfce_values = []
                for iteration in iterator:
                    vals = {hemi: {args.metric: t_vals[f"{hemi}__{args.metric}"][iteration]} for hemi in HEMIS}
                    tfce_values.append(calc_tfce_values(vals, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e))
                # tfce_values = [
                #     calc_tfce_values(vals, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e) for vals in
                #     iterator
                # ]
                return tfce_values

        n_per_job = math.ceil(args.n_permutations_group_level / args.n_jobs)
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


def create_masks(args):
    print("Creating gifti results masks")
    p_values_path = os.path.join(permutation_results_dir(args), f"p_values{get_hparam_suffix(args)}.p")
    masks_path = os.path.join(os.path.dirname(p_values_path), f"masks{get_hparam_suffix(args)}")
    os.makedirs(masks_path, exist_ok=True)
    p_values_gifti_path = os.path.join(os.path.dirname(p_values_path), "p_values_gifti")
    os.makedirs(p_values_gifti_path, exist_ok=True)

    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    log_10_p_values = copy.deepcopy(p_values)
    log_10_p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    log_10_p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    for hemi in HEMIS:
        path_out = os.path.join(permutation_results_dir(args),
                                f"p_values{get_hparam_suffix(args)}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(log_10_p_values[hemi], path_out)

    # tfce values
    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
    tfce_values = pickle.load(open(tfce_values_path, "rb"))

    for hemi in HEMIS:
        path_out = os.path.join(permutation_results_dir(args),
                                f"tfce_values{get_hparam_suffix(args)}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(tfce_values[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC], path_out)

    # p value masks
    masks = copy.deepcopy(p_values)
    for hemi in HEMIS:
        masks[hemi][p_values[hemi] < args.p_value_threshold] = 1
        masks[hemi][p_values[hemi] >= args.p_value_threshold] = 0
        masks[hemi][np.isnan(p_values[hemi])] = 0
        masks[hemi] = masks[hemi].astype(np.uint8)

    path_out = os.path.join(masks_path, f"p_values{get_hparam_suffix(args)}_thresh_{args.p_value_threshold}.p")
    pickle.dump(masks, open(path_out, mode='wb'))

    edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)
    for hemi in HEMIS:
        print(hemi)
        results = calc_clusters(masks[hemi], threshold=1e-8, edge_lengths=edge_lengths[hemi], return_clusters=True)
        clusters = results['clusters']
        clusters.sort(key=len, reverse=True)
        for i, cluster in enumerate(clusters[:10]):
            print(f"{i}: Cluster of size {len(cluster)}")
            cluster_map = np.repeat(np.nan, log_10_p_values[hemi].shape)
            cluster_map[list(cluster)] = log_10_p_values[hemi][list(cluster)]

            fname = f"thresh_{args.p_value_threshold}_{FS_HEMI_NAMES[hemi]}_cluster_{i}.gii"
            path_out = os.path.join(p_values_gifti_path, fname)
            export_to_gifti(cluster_map, path_out)

            cluster_mask = {h: np.zeros_like(cluster_map, dtype=np.uint8) for h in HEMIS}
            cluster_mask[hemi][list(cluster)] = 1
            path_out = os.path.join(masks_path, f"p_values_thresh_{args.p_value_threshold}_{hemi}_cluster_{i}.p")
            pickle.dump(cluster_mask, open(path_out, mode='wb'))


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

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-permutations-group-level", type=int, default=10000)

    parser.add_argument("--metric", type=str, default=METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, exist_ok=True)
    args = get_args()

    create_null_distribution(args)
    calc_test_statistics(args)
    create_masks(args)
