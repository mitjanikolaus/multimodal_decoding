import argparse
import copy
import math
import warnings

import numpy as np
from joblib import Parallel, delayed
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle

from nilearn.surface import surface
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns

from analyses.ridge_regression_decoding import FEATS_SELECT_DEFAULT, get_default_features, FEATURE_COMBINATION_CHOICES
from analyses.searchlight import SEARCHLIGHT_OUT_DIR
from utils import RESULTS_DIR, SUBJECTS, HEMIS

DEFAULT_N_JOBS = 10

METRIC_CAPTIONS = 'captions'
METRIC_IMAGES = 'images'
METRIC_DIFF_CAPTIONS = 'captions_agno - captions_specific'
METRIC_DIFF_IMAGES = 'imgs_agno - imgs_specific'
METRIC_MIN_DIFF_BOTH_MODALITIES = 'min(captions_agno - captions_specific, imgs_agno - imgs_specific)'
METRIC_MIN_ALT = 'min_alternative'

METRIC_CODES = {
    METRIC_MIN_DIFF_BOTH_MODALITIES: 0,
    METRIC_DIFF_CAPTIONS: 1,
    METRIC_DIFF_IMAGES: 2,
    METRIC_MIN_ALT: 3,
}

BASE_METRICS = ["test_captions", "test_images"]
TEST_METRICS = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES]
CHANCE_VALUES = {
    METRIC_CAPTIONS: 0.5,
    METRIC_IMAGES: 0.5,
    METRIC_DIFF_IMAGES: 0,
    METRIC_DIFF_CAPTIONS: 0,
    METRIC_MIN_DIFF_BOTH_MODALITIES: 0,
    METRIC_MIN_ALT: 0,
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


def smooth_surface_data(matrix, surf_data,
                        iterations=1,
                        vertex_weights=None,
                        return_vertex_weights=False,
                        center_surround_knob=0,
                        match='sum'):
    """Smooth values along the surface.

    Parameters
    ----------
    surface : Surface-like
        The surface on which the data `surf_data` are to be smoothed.
    surf_data : array-like
        The array of values at each vertex that is being smoothed. This may
        either be a vector of length `n` or a matrix with `n` rows.  In the case
        of fMRI data, `n` could be the number of timepoints. Each column is
        smoothed independently.
    iterations : :obj:`int`, optional
        The number of times to repeat the smoothing operation (it must be a positive value).
        Defaults to 1
    distance_weights : :obj:`bool`, optional
        Whether to add distance-based weighting to the smoothing. With such
        weights, the value calculated for each vertex at each iteration is the
        weighted sum of neighboring vertices where the weight on each neighbor
        is the inverses of the distances to it.
    vertex_weights : array-like or None, optional
        A vector of weights, one per vertex. These weights are normalized and
        applied to the smoothing after the application of center-surround
        weights.
    return_vector_weights : :obj:`bool`, optional
        If `True` then `(smoothed_data, smoothed_vertex_weights)` are returned.
        The default is `False`.
    center_surround_knob : :obj:`float`, optional
        The relative weighting of the center and the surround in each iteration
        of the smoothing. If the value of the knob is `k`, then the total weight
        of vertices that are neighbors of a given vertex (the vertex's surround)
        is `2**k` times the weight of the vertex itself (the center). A value of
        0 (the default) means that, in each smoothing iteration, each vertex is
        updated with the average of its value and the average value of its
        neighbors. A value of `-inf` results in no smoothing because the entire
        weight is on the center, so each vertex is updated with its own value. A
        value of `inf` results in each vertex being updated with the average of
        its neighbors without including its own value.
    match : { 'sum' | 'mean' | 'var' | 'dist' | None }, optional
        What properties of the input data should be matched in the output data.
        `None` indicates that the smoothed output should be
        returned without transformation. If the value is `'sum'`, then the
        output is rescaled to have the same sum as `surf_data`. If the value is
        `'mean'`, then the output is shifted to match the mean of the input. If
        the value is `'var'` or `'std'`, then the variance of the output is
        matched. Finally, if the value is `'dist'`, then the mean and the
        variance are matched. Default is `'sum'`

    Returns
    -------
    surf_data_smooth : array
        The array of smoothed values at each vertex.
    Examples
    -------
    >>> from nilearn import datasets, surface, plotting
    >>> fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
    >>> white_left = surface.load_surf_mesh(fsaverage.white_left)
    >>> curv = surface.load_surf_data(fsaverage.curv_left)
    >>> curv_smooth = surface.smooth_surface_data(surface=white_left, surf_data=curv, iterations=50)
    >>> plotting.plot_surf(white_left, surf_map = curv_smooth)
    """
    from scipy.sparse import diags
    # First, calculate the center and surround weights for the
    # center-surround knob.
    center_weight = 1 / (1 + np.exp2(-center_surround_knob))
    surround_weight = 1 - center_weight
    if surround_weight == 0:
        # There's nothing to do in this case.
        return np.array(surf_data)

    # If there are vertex weights, get them ready.
    if vertex_weights:
        w = np.array(vertex_weights)
        w /= np.sum(w)
    else:
        w = np.ones(matrix.shape[0])
    # We need to normalize the matrix columns, and we can do this now by
    # normalizing everything but the diagonal to the surround weight, then
    # adding the center weight along the diagonal.
    colsums = matrix.sum(axis=1)
    colsums = np.asarray(colsums).flatten()
    matrix = matrix.multiply(surround_weight / colsums[:, None])

    # Add in the diagonal.
    matrix.setdiag(center_weight)
    # Run the iterations of smoothing.
    data = surf_data
    for _ in range(iterations):
        data[np.isneginf(data)] = 1e-8  # numerical stability
        if np.sum(np.isnan(data)) > 0:
            data[~np.isnan(data)] = matrix.A[~np.isnan(data)][:, ~np.isnan(data)].dot(data[~np.isnan(data)])
        else:
            data = matrix.dot(data)
    # Convert back into numpy array.
    data = np.reshape(np.asarray(data), np.shape(surf_data))
    # Rescale it if needed.
    if match == 'sum':
        sum0 = np.nansum(surf_data, axis=0)
        sum1 = np.nansum(data, axis=0)
        data = data * (sum0 / sum1)
    elif match == 'mean':
        mu0 = np.nansum(surf_data, axis=0)
        mu1 = np.nansum(data, axis=0)
        data = data + (mu0 - mu1)
    elif match in ('var', 'std', 'variance', 'stddev', 'sd'):
        std0 = np.nanstd(surf_data, axis=0)
        std1 = np.nanstd(data, axis=0)
        mu1 = np.nanmean(data, axis=0)
        data = (data - mu1) * (std0 / std1) + mu1
    elif match in ('dist', 'meanvar', 'meanstd', 'meansd'):
        std0 = np.nanstd(surf_data, axis=0)
        std1 = np.nanstd(data, axis=0)
        mu0 = np.nanmean(surf_data, axis=0)
        mu1 = np.nanmean(data, axis=0)
        data = (data - mu1) * (std0 / std1) + mu0
    elif match is not None:
        raise ValueError(f"invalid match argument: {match}")
    if return_vertex_weights:
        w /= np.sum(w)
        return (data, w)
    else:
        return data


# def get_adj_matrices(resolution):
#     adj_path = os.path.join(SEARCHLIGHT_OUT_DIR, "adjacency_matrices", resolution, "adjacency.p")
#     if not os.path.isfile(adj_path):
#         print("Computing adjacency matrices.")
#         adjacency_matrices = dict()
#         for hemi in HEMIS:
#             os.makedirs(os.path.dirname(adj_path), exist_ok=True)
#             fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
#             surface_infl = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])
#
#             adjacency_matrix = compute_adjacency_matrix(surface_infl, values='euclidean')
#
#             adjacency_matrices[hemi] = adjacency_matrix
#         pickle.dump(adjacency_matrices, open(adj_path, 'wb'))
#     else:
#         adjacency_matrices = pickle.load(open(adj_path, 'rb'))
#
#     return adjacency_matrices


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


def calc_image_t_values(data, popmean):
    enough_data = (~np.isnan(data)).sum(axis=0) > 2  # at least 3 datapoints
    return np.array([
        stats.ttest_1samp(x[~np.isnan(x)], popmean=popmean, alternative="greater")[0] if ed else np.nan for x, ed
        in zip(data.T, enough_data)]
    )


def calc_t_values(per_subject_scores):
    t_values = {hemi: dict() for hemi in HEMIS}
    for hemi in HEMIS:
        for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, METRIC_IMAGES, METRIC_CAPTIONS]:
            data = np.array([per_subject_scores[subj][hemi][metric] for subj in SUBJECTS])
            popmean = CHANCE_VALUES[metric]
            t_values[hemi][metric] = calc_image_t_values(data, popmean)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
                (t_values[hemi][METRIC_DIFF_CAPTIONS], t_values[hemi][METRIC_DIFF_IMAGES]),
                axis=0)

            t_values[hemi][METRIC_MIN_ALT] = np.nanmin(
                (
                    t_values[hemi][METRIC_DIFF_CAPTIONS],
                    t_values[hemi][METRIC_DIFF_IMAGES],
                    t_values[hemi][METRIC_IMAGES],
                    t_values[hemi][METRIC_CAPTIONS]),
                axis=0)

    return t_values


def load_per_subject_scores(model, features, resolution, mode, alpha):
    per_subject_scores = {subj: dict() for subj in SUBJECTS}

    results_regex = os.path.join(SEARCHLIGHT_OUT_DIR,
                                 f'train/{model}/{features}/*/{resolution}/*/{mode}/alpha_{str(alpha)}.p')
    paths_mod_agnostic = np.array(sorted(glob(results_regex)))
    paths_mod_specific_captions = np.array(sorted(glob(results_regex.replace('train/', 'train_captions/'))))
    paths_mod_specific_images = np.array(sorted(glob(results_regex.replace('train/', 'train_images/'))))
    assert len(paths_mod_agnostic) == len(paths_mod_specific_images) == len(paths_mod_specific_captions)

    for path_agnostic, path_caps, path_imgs in tqdm(zip(paths_mod_agnostic, paths_mod_specific_captions,
                                                        paths_mod_specific_images), total=len(paths_mod_agnostic)):
        hemi = os.path.dirname(path_agnostic).split("/")[-2]
        subject = os.path.dirname(path_agnostic).split("/")[-4]

        results_agnostic = pickle.load(open(path_agnostic, 'rb'))
        scores_agnostic = results_agnostic['scores']
        scores_captions = pickle.load(open(path_caps, 'rb'))['scores']
        scores_images = pickle.load(open(path_imgs, 'rb'))['scores']

        nan_locations = results_agnostic['nan_locations']
        scores = process_scores(scores_agnostic, scores_captions, scores_images, nan_locations)
        # print({n: round(np.nanmean(score), 4) for n, score in scores.items()})
        # print({f"{n}_max": round(np.nanmax(score), 2) for n, score in scores.items()})
        # print("")

        per_subject_scores[subject][hemi] = scores
    return per_subject_scores


def run(args):
    t_values_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
                                 "t_values.p")
    if not os.path.isfile(t_values_path):
        os.makedirs(os.path.dirname(t_values_path), exist_ok=True)
        print(f"Calculating t-values")
        per_subject_scores = load_per_subject_scores(args.model, args.features, args.resolution, args.mode,
                                                     args.l2_regularization_alpha)
        t_values = calc_t_values(per_subject_scores)

        pickle.dump(t_values, open(t_values_path, 'wb'))
    else:
        t_values = pickle.load(open(t_values_path, 'rb'))

    if args.smoothing_iterations > 0:
        print("smoothing")
        fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
        smooth_t_values = copy.deepcopy(t_values)
        for hemi in HEMIS:
            surface_infl = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])
            # Calculate the adjacency matrix either weighting by inverse distance or not weighting (ones)
            distance_weights = True
            values = 'inveuclidean' if distance_weights else 'ones'
            adj_matrix = compute_adjacency_matrix(surface_infl, values=values)
            smooth_t_values[hemi][args.metric] = smooth_surface_data(
                adj_matrix, t_values[hemi][args.metric], match=None,
                iterations=args.smoothing_iterations
            )
        t_values = smooth_t_values
        t_values_smooth_path = os.path.join(
            SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
            f"t_values_metric_{METRIC_CODES[args.metric]}_smoothed_{args.smoothing_iterations}.p")
        pickle.dump(smooth_t_values, open(t_values_smooth_path, 'wb'))

    print("calculating tfce..")
    edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)
    tfce_values = calc_tfce_values(t_values, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e)

    tfce_values_path = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
        f"tfce_values_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p")
    pickle.dump(tfce_values, open(tfce_values_path, "wb"))

    null_distribution_tfce_values_file = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
        args.resolution,
        args.mode,
        f"tfce_values_null_distribution_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
    )

    print("loading null distribution test statistic: ", null_distribution_tfce_values_file)
    null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))

    max_test_statistic_distr = {
        hemi: sorted([np.nanmax(n[hemi][args.metric]) for n in null_distribution_tfce_values])
        for hemi in HEMIS
    }

    significance_cutoffs = {hemi: np.quantile(max_test_statistic_distr[hemi], 0.95) for hemi in HEMIS}
    print(f"{len(null_distribution_tfce_values)} permutations")
    print(f"cluster test statistic significance cutoff for p<0.05 (left hemi): {significance_cutoffs['left']}")
    print(f"cluster test statistic significance cutoff for p<0.05 (right hemi): {significance_cutoffs['right']}")

    p_values = {hemi: np.zeros_like(t_vals[args.metric]) for hemi, t_vals in
                t_values.items()}
    for hemi in HEMIS:
        print(f"{hemi} hemi largest test statistic values: ",
              sorted([t for t in tfce_values[hemi][args.metric]], reverse=True)[:10])
        for vertex in np.argwhere(tfce_values[hemi][args.metric] > 0)[:, 0]:
            test_stat = tfce_values[hemi][args.metric][vertex]
            value_indices = np.argwhere(max_test_statistic_distr[hemi] > test_stat)
            if len(value_indices) > 0:
                p_value = 1 - value_indices[0].item() / len(null_distribution_tfce_values)
            else:
                p_value = 1 - (len(null_distribution_tfce_values) - 1) / (len(null_distribution_tfce_values))
            p_values[hemi][vertex] = p_value

    print(f"smallest p value (left): {np.min(p_values['left'][p_values['left'] > 0]):.4f}")
    print(f"smallest p value (right): {np.min(p_values['right'][p_values['right'] > 0]):.4f}")

    p_values_path = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
        f"p_values_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
    )

    pickle.dump(p_values, open(p_values_path, mode='wb'))


def load_null_distr_per_subject_scores(args):
    alpha = args.l2_regularization_alpha
    results_regex = os.path.join(
        SEARCHLIGHT_OUT_DIR,
        f'train/{args.model}/{args.features}/*/{args.resolution}/*/{args.mode}/alpha_{str(alpha)}.p'
    )
    per_subject_scores_null_distr = []
    paths_mod_agnostic = np.array(sorted(glob(results_regex)))
    paths_mod_specific_captions = np.array(sorted(glob(results_regex.replace('train/', 'train_captions/'))))
    paths_mod_specific_images = np.array(sorted(glob(results_regex.replace('train/', 'train_images/'))))
    assert len(paths_mod_agnostic) == len(paths_mod_specific_images) == len(paths_mod_specific_captions)

    for path_agnostic, path_caps, path_imgs in tqdm(zip(paths_mod_agnostic, paths_mod_specific_captions,
                                                        paths_mod_specific_images), total=len(paths_mod_agnostic)):
        hemi = os.path.dirname(path_agnostic).split("/")[-2]
        subject = os.path.dirname(path_agnostic).split("/")[-4]

        results_agnostic = pickle.load(open(path_agnostic, 'rb'))
        nan_locations = results_agnostic['nan_locations']

        legacy_null_distr_filename = f"alpha_{str(alpha)}_null_distribution.p"
        if os.path.isfile(os.path.join(os.path.dirname(path_agnostic), legacy_null_distr_filename)):
            print("loading null distribution files from legacy format..")
            null_distribution_agnostic = pickle.load(
                open(os.path.join(os.path.dirname(path_agnostic), legacy_null_distr_filename), 'rb'))

            null_distribution_images = pickle.load(
                open(os.path.join(os.path.dirname(path_imgs), legacy_null_distr_filename), 'rb'))

            null_distribution_captions = pickle.load(
                open(os.path.join(os.path.dirname(path_caps), legacy_null_distr_filename), 'rb'))

            for i, (distr, distr_caps, distr_imgs) in enumerate(zip(null_distribution_agnostic,
                                                                    null_distribution_captions,
                                                                    null_distribution_images)):
                if len(per_subject_scores_null_distr) <= i:
                    per_subject_scores_null_distr.append({subj: dict() for subj in SUBJECTS})
                scores = process_scores(distr, distr_caps, distr_imgs, nan_locations)
                per_subject_scores_null_distr[i][subject][hemi] = scores
        else:
            def load_null_distr_scores(base_path):
                scores_dir = os.path.join(base_path, "null_distr")
                score_paths = sorted(list(glob(os.path.join(scores_dir, "*.p"))))
                last_idx = int(os.path.basename(score_paths[-1])[:-2])
                assert last_idx == len(score_paths) - 1, last_idx
                scores = [pickle.load(open(score_path, "rb")) for score_path in score_paths]
                return scores

            null_distribution_agnostic = load_null_distr_scores(os.path.dirname(path_agnostic))
            null_distribution_images = load_null_distr_scores(os.path.dirname(path_imgs))
            null_distribution_captions = load_null_distr_scores(os.path.dirname(path_caps))

            num_permutations = len(null_distribution_agnostic[0])
            for i in range(num_permutations):
                distr = [null_distr[i] for null_distr in null_distribution_agnostic]
                distr_caps = [null_distr[i] for null_distr in null_distribution_captions]
                distr_imgs = [null_distr[i] for null_distr in null_distribution_images]
                if len(per_subject_scores_null_distr) <= i:
                    per_subject_scores_null_distr.append({subj: dict() for subj in SUBJECTS})
                scores = process_scores(distr, distr_caps, distr_imgs, nan_locations)
                per_subject_scores_null_distr[i][subject][hemi] = scores
    return per_subject_scores_null_distr


def calc_t_values_null_distr(args):
    per_subject_scores_null_distr = load_null_distr_per_subject_scores(args)

    def shuffle_and_calc_t_values(per_subject_scores, proc_id, n_iters_per_job):
        job_t_vals = []
        iterator = tqdm(range(n_iters_per_job)) if proc_id == 0 else range(n_iters_per_job)
        for _ in iterator:
            t_values = {hemi: dict() for hemi in HEMIS}
            for hemi in HEMIS:
                for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, METRIC_IMAGES, METRIC_CAPTIONS]:
                    random_idx = np.random.choice(len(per_subject_scores), size=len(SUBJECTS))
                    data = np.array(
                        [per_subject_scores[idx][subj][hemi][metric] for idx, subj in
                         zip(random_idx, SUBJECTS)])
                    popmean = CHANCE_VALUES[metric]
                    t_values[hemi][metric] = calc_image_t_values(data, popmean)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
                        (t_values[hemi][METRIC_DIFF_CAPTIONS], t_values[hemi][METRIC_DIFF_IMAGES]),
                        axis=0)
                    t_values[hemi][METRIC_MIN_ALT] = np.nanmin(
                        (
                            t_values[hemi][METRIC_DIFF_CAPTIONS],
                            t_values[hemi][METRIC_DIFF_IMAGES],
                            t_values[hemi][METRIC_IMAGES],
                            t_values[hemi][METRIC_CAPTIONS]),
                        axis=0)

            job_t_vals.append(t_values)
        return job_t_vals

    n_iters_per_job = math.ceil(args.n_permutations_group_level / args.n_jobs)
    n_iters_last_job = n_iters_per_job
    if args.n_permutations_group_level % args.n_jobs != 0:
        n_iters_last_job = (n_iters_per_job * args.n_jobs) - args.n_permutations_group_level
    print(f"n iters per job: {n_iters_per_job} (last job: {n_iters_last_job})")
    all_t_vals = Parallel(n_jobs=args.n_jobs)(
        delayed(shuffle_and_calc_t_values)(
            per_subject_scores_null_distr.copy(),
            id,
            n_iters_per_job if not id == args.n_jobs - 1 else n_iters_last_job,
        )
        for id in range(args.n_jobs)
    )
    all_t_vals = np.concatenate(all_t_vals)

    return all_t_vals


def create_null_distribution(args):
    t_values_null_distribution_path = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
        args.resolution,
        args.mode, f"t_values_null_distribution.p"
    )
    if not os.path.isfile(t_values_null_distribution_path):
        print(f"Calculating t-values: null distribution")
        t_values_null_distribution = calc_t_values_null_distr(args)
        os.makedirs(os.path.dirname(t_values_null_distribution_path), exist_ok=True)
        pickle.dump(t_values_null_distribution, open(t_values_null_distribution_path, 'wb'))
    else:
        t_values_null_distribution = pickle.load(open(t_values_null_distribution_path, 'rb'))

    if args.smoothing_iterations > 0:
        smooth_t_values_null_distribution_path = os.path.join(
            SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
            args.resolution,
            args.mode,
            f"t_values_null_distribution_metric_{METRIC_CODES[args.metric]}_smoothed_{args.smoothing_iterations}.p"
        )
        if not os.path.isfile(smooth_t_values_null_distribution_path):
            print("smoothing for null distribution")

            def smooth_t_values(t_values, proc_id):
                smooth_t_vals = []
                fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
                surface_infl = {hemi: surface.load_surf_mesh(fsaverage[f"infl_{hemi}"]) for hemi in HEMIS}
                # Calculate the adjacency matrix either weighting by inverse distance or not weighting (ones)
                distance_weights = True
                values = 'inveuclidean' if distance_weights else 'ones'
                adj_matrices = {hemi: compute_adjacency_matrix(surface_infl[hemi], values=values) for hemi in HEMIS}
                iterator = tqdm(t_values) if proc_id == 0 else t_values
                for t_vals in iterator:
                    for hemi in HEMIS:
                        smoothed = smooth_surface_data(
                            adj_matrices[hemi],
                            t_vals[hemi][args.metric],
                            match=None,
                            iterations=args.smoothing_iterations
                        )
                        t_vals[hemi][args.metric] = smoothed
                    smooth_t_vals.append(t_vals)
                return smooth_t_vals

            n_per_job = math.ceil(len(t_values_null_distribution) / args.n_jobs)
            all_smooth_t_vals = Parallel(n_jobs=args.n_jobs)(
                delayed(smooth_t_values)(
                    t_values_null_distribution[id * n_per_job:(id + 1) * n_per_job],
                    id,
                )
                for id in range(args.n_jobs)
            )
            smooth_t_values_null_distribution = np.concatenate(all_smooth_t_vals)
            pickle.dump(smooth_t_values_null_distribution, open(smooth_t_values_null_distribution_path, 'wb'))

        t_values_null_distribution = pickle.load(open(smooth_t_values_null_distribution_path, 'rb'))

    tfce_values_null_distribution_path = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
        args.resolution,
        args.mode,
        f"tfce_values_null_distribution_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
    )
    if not os.path.isfile(tfce_values_null_distribution_path):
        print(f"Calculating tfce values for null distribution")

        edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)

        def tfce_values_job(t_values, edge_lengths, proc_id):
            iterator = tqdm(t_values) if proc_id == 0 else t_values
            tfce_values = [
                calc_tfce_values(vals, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e) for vals in
                iterator
            ]
            return tfce_values

        n_per_job = math.ceil(len(t_values_null_distribution) / args.n_jobs)
        tfce_values = Parallel(n_jobs=args.n_jobs)(
            delayed(tfce_values_job)(
                t_values_null_distribution[id * n_per_job:(id + 1) * n_per_job],
                edge_lengths.copy(),
                id,
            )
            for id in range(args.n_jobs)
        )
        tfce_values = np.concatenate(tfce_values)

        pickle.dump(tfce_values, open(tfce_values_null_distribution_path, 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='vilt')
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default='fsaverage7')
    parser.add_argument("--mode", type=str, default='n_neighbors_100')

    parser.add_argument("--smoothing-iterations", type=int, default=0)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-permutations-group-level", type=int, default=1000)

    parser.add_argument("--metric", type=str, default=METRIC_MIN_ALT)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.features = get_default_features(args.model) if args.features == FEATS_SELECT_DEFAULT else args.features

    create_null_distribution(args)
    run(args)
