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
from nilearn.surface import surface, load_surface, load_surf_mesh
from scipy import stats
from scipy.sparse import csr_matrix
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


def smooth_surface_data(surface, surf_data,
                        iterations=1,
                        distance_weights=False,
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
    # Calculate the adjacency matrix either weighting by inverse distance or not weighting (ones)
    values = 'inveuclidean' if distance_weights else 'ones'
    matrix = compute_adjacency_matrix(surface, values=values)

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
    # Run the iterations of smooothing.
    data = surf_data
    for _ in range(iterations):
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


def get_adj_matrices(resolution):
    adj_path = os.path.join(SEARCHLIGHT_OUT_DIR, "adjacency_matrices", resolution, "adjacency.p")
    if not os.path.isfile(adj_path):
        print("Computing adjacency matrices.")
        adjacency_matrices = dict()
        for hemi in HEMIS:
            os.makedirs(os.path.dirname(adj_path), exist_ok=True)
            fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
            surface_infl = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])

            adjacency_matrix = compute_adjacency_matrix(surface_infl, values='euclidean')

            adjacency_matrices[hemi] = adjacency_matrix
        pickle.dump(adjacency_matrices, open(adj_path, 'wb'))
    else:
        adjacency_matrices = pickle.load(open(adj_path, 'rb'))

    return adjacency_matrices


def calc_tfce_values(t_values, adjacency_matrices, resolution, h=2, e=1, dh="auto"):
    tfce_values = dict()
    fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
    for hemi in HEMIS:
        values = t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES]
        max_score = np.nanmax(values)
        step = max_score / 100 if dh == "auto" else dh

        score_threshs = np.arange(step, max_score + step, step)

        surface_infl = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])

        edges = np.vstack([surface_infl.faces[:, [0, 1]],
                           surface_infl.faces[:, [0, 2]],
                           surface_infl.faces[:, [1, 2]]])

        edges = np.unique(edges, axis=0)
        coords = surface_infl.coordinates
        lengths = np.sqrt(np.sum((coords[edges[:, 0]] - coords[edges[:, 1]]) ** 2, axis=1))
        edge_lengths_dict = {(e[0], e[1]): l for e, l in zip(edges, lengths)}
        edge_lengths_dict.update({(e[1], e[0]): l for e, l in zip(edges, lengths)})

        tfce_values[hemi] = {METRIC_MIN_DIFF_BOTH_MODALITIES: np.zeros_like(values)}

        for score_thresh in score_threshs:
            # values[values < score_thresh] = 0
            # print("remaining: ", np.sum(values > 0))

            clusters_dict = calc_clusters(values, adjacency_matrices[hemi],
                                          score_thresh,
                                          edge_lengths_dict,
                                          return_clusters=True,
                                          return_cluster_edge_lengths=True,
                                          return_agg_t_values=False,
                                          min_cluster_size=2)
            clusters = clusters_dict["clusters"]
            cluster_extends = np.array(clusters_dict["cluster_edge_lengths"])

            cluster_tfces = (cluster_extends ** e) * (score_thresh ** h)
            for cluster, cluster_tfce in zip(clusters, cluster_tfces):
                tfce_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES][list(cluster)] += step * cluster_tfce

        from nilearn import plotting
        # plotting.plot_surf_stat_map(
        #     surface_infl,
        #     t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES],
        #     hemi=hemi,
        #     view="lateral",
        #     bg_map=fsaverage[f"sulc_{hemi}"],
        #     colorbar=True,
        # )
    return tfce_values


def calc_clusters(scores, adj, t_value_threshold, edge_lengths=None, return_clusters=False,
                  return_cluster_edge_lengths=False, return_agg_t_values=True, min_cluster_size=1):
    clusters = []
    cluster_t_values = []
    cluster_edge_lengths = []
    scores_thresholded = scores > t_value_threshold

    start_locations = list(np.argwhere(scores_thresholded)[:, 0])
    # cluster_maps = np.zeros_like(scores)
    while len(start_locations) > 0:
        idx = start_locations[0]
        cluster = {idx}
        c_edge_lengths = []
        checked = {idx}

        def expand_neighbors(start_idx, adj):
            neighbors = np.argwhere(adj[start_idx] > 0)[:, 1]
            for neighbor in neighbors:
                if not neighbor in checked:
                    checked.add(neighbor)
                    if scores_thresholded[neighbor]:
                        cluster.add(neighbor)
                        if return_cluster_edge_lengths:
                            c_edge_lengths.append(edge_lengths[(start_idx, neighbor)])
                        expand_neighbors(neighbor, adj)

        expand_neighbors(idx, adj)
        for id in cluster:
            start_locations.remove(id)

        if len(cluster) > min_cluster_size:
            if return_agg_t_values:
                t_value_cluster = np.sum(scores[list(cluster)])
                cluster_t_values.append(t_value_cluster)
            if return_clusters:
                clusters.append(cluster)
            if return_cluster_edge_lengths:
                cluster_edge_lengths.append(np.sum(c_edge_lengths))
        # cluster_maps[list(cluster)] = scores[list(cluster)]

    # fill non-cluster locations with zeros
    # cluster_maps[cluster_maps == 0] = 0

    result_dict = dict()
    if return_clusters:
        result_dict['clusters'] = clusters
    if return_agg_t_values:
        result_dict['agg_t_values'] = cluster_t_values
    if return_cluster_edge_lengths:
        result_dict['cluster_edge_lengths'] = cluster_edge_lengths
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
        t_vals = dict()
        for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]:
            data = np.array([per_subject_scores[subj][hemi][metric] for subj in SUBJECTS])
            popmean = CHANCE_VALUES[metric]
            t_vals[metric] = calc_image_t_values(data, popmean)

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

    adjacency_matrices = get_adj_matrices(args.resolution)

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

    # fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    # surface_infl = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])
    #
    # from nilearn import plotting
    # plotting.plot_surf_stat_map(
    #     surface_infl,
    #     t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES],
    #     hemi=hemi,
    #     view="lateral",
    #     bg_map=fsaverage[f"sulc_{hemi}"],
    #     colorbar=True,
    # )

    print("smoothing")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    for hemi in HEMIS:
        surface_infl = surface.load_surf_mesh(fsaverage[f"infl_{hemi}"])
        t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = smooth_surface_data(surface_infl, t_values[hemi][
            METRIC_MIN_DIFF_BOTH_MODALITIES], distance_weights=True, match=None)

    if args.tfce:
        print("calculating tfce..")
        tfce_values = calc_tfce_values(t_values, adjacency_matrices, args.resolution)
        t_values = tfce_values

    print(f"calculating clusters for threshold t>{args.t_value_threshold}")
    clusters, cluster_t_values = dict(), dict()
    for hemi in HEMIS:
        clusters_dict = calc_clusters(
            t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES],
            adjacency_matrices[hemi],
            args.t_value_threshold,
            return_clusters=True
        )
        clusters[hemi] = clusters_dict["clusters"]
        cluster_t_values[hemi] = clusters_dict["agg_t_values"]

    filename = f"clusters_null_distribution_t_thresh_{args.t_value_threshold}{'_tfce' if args.tfce else ''}.p"
    clusters_null_distribution_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, features,
                                                   args.resolution,
                                                   args.mode, filename)
    clusters_null_distribution = pickle.load(open(clusters_null_distribution_path, 'rb'))
    if isinstance(clusters_null_distribution[0], tuple):
        clusters_null_distribution = [c[1] for c in clusters_null_distribution]

    max_cluster_t_value_distr = {
        hemi: sorted([
            np.max(c[hemi]) if len(c[hemi]) > 0 else 0 for c in clusters_null_distribution
        ]) for hemi in HEMIS
    }

    significance_cutoffs = {hemi: np.quantile(max_cluster_t_value_distr[hemi], 0.95) for hemi in HEMIS}
    print(f"{len(clusters_null_distribution)} permutations")
    print(f"cluster t-value significance cutoff for p<0.05 (left hemi): {significance_cutoffs['left']}")
    print(f"cluster t-value significance cutoff for p<0.05 (right hemi): {significance_cutoffs['right']}")

    p_values_cluster = {hemi: np.zeros_like(t_vals[METRIC_MIN_DIFF_BOTH_MODALITIES]) for hemi, t_vals in t_values.items()}
    for hemi in HEMIS:
        print(f"{hemi} hemi largest cluster t-values: ", sorted([t for t in cluster_t_values[hemi]], reverse=True)[:10])
        for cluster, t_val in zip(clusters[hemi], cluster_t_values[hemi]):
            value_indices = np.argwhere(max_cluster_t_value_distr[hemi] > t_val)
            p_value = 1 - value_indices[0] / len(clusters_null_distribution) if len(value_indices) > 0 else 1 - (
                    len(clusters_null_distribution) - 1) / (len(clusters_null_distribution))
            p_values_cluster[hemi][list(cluster)] = p_value

    # transform to plottable magnitudes:
    p_values_cluster['left'][p_values_cluster['left'] > 0] = -np.log10(
        p_values_cluster['left'][p_values_cluster['left'] > 0])
    p_values_cluster['right'][p_values_cluster['right'] > 0] = -np.log10(
        p_values_cluster['right'][p_values_cluster['right'] > 0])

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

    parser.add_argument("--tfce", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
