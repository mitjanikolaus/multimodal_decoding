import copy
import os
import pickle

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.surface import surface

from analyses.decoding.searchlight.searchlight import get_adjacency_matrix
from utils import export_to_gifti, HEMIS, FS_HEMI_NAMES


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


def create_results_cluster_masks(values, results_dir, hparam_suffix, metric, resolution, radius, n_neighbors, threshold):
    t_values_path = os.path.join(results_dir, "t_values.p")
    t_values = pickle.load(open(t_values_path, "rb"))

    p_values_path = os.path.join(results_dir, f"p_values{hparam_suffix}.p")
    p_values = pickle.load(open(p_values_path, "rb"))

    edge_lengths = get_edge_lengths_dicts_based_on_edges(resolution)
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")

    results_maps_path = os.path.join(results_dir, f"results_maps")
    masks_path = os.path.join(os.path.dirname(p_values_path), f"masks{hparam_suffix}")
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
            vertex_max_t_value = cluster[np.nanargmax(t_values[hemi][metric][cluster])]
            max_t_value = t_values[hemi][metric][vertex_max_t_value]
            print(f"Max t-value: {max_t_value:.2f}", end=" | ")
            coords = mesh.coordinates[vertex_max_t_value]
            print(f"Coordinates (max t-value): {coords}")
            clusters_df.append({
                "hemi": hemi, "id": i, "location": "", "size": len(cluster),
                "max t-value": max_t_value,
                "p-value": '{:.0e}'.format(p_values[hemi][vertex_max_t_value]),
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


def calc_significance_cutoff(null_distribution_tfce_values, metric, p_value_threshold=0.05, multiple_comparisons_control=True):
    if multiple_comparisons_control:
        null_distr = np.sort([
            np.nanmax(np.concatenate((n[HEMIS[0]][metric], n[HEMIS[1]][metric])))
            for n in null_distribution_tfce_values
        ])
    else:
        print('not using multiple comparisons control')
        null_distr = np.sort(np.array([
            np.concatenate((n[HEMIS[0]][metric], n[HEMIS[1]][metric]))
            for n in null_distribution_tfce_values
        ]).flatten())
        print(null_distr.shape)

    print(f"{len(null_distribution_tfce_values)} permutations")
    print(f"null distr size: {len(null_distr)}")
    if p_value_threshold == 1 / len(null_distribution_tfce_values):
        significance_cutoff = np.max(null_distr)
    else:
        significance_cutoff = np.quantile(null_distr, 1 - p_value_threshold, method='closest_observation')

    for thresh in [0.05, 1e-2, 1e-3, 1e-4]:
        if thresh == 1/len(null_distribution_tfce_values):
            val = np.max(null_distr)
        else:
            val = np.quantile(null_distr, 1 - thresh, method='closest_observation')
        print(f"(info) cluster test statistic significance cutoff for p<{thresh}: {val:.2f}")

    print(f"using cluster test statistic significance cutoff for p<{p_value_threshold}: {significance_cutoff:.3f}")

    return significance_cutoff, null_distr


def create_masks(results_dir, metric, p_value_threshold, tfce_value_threshold, hparam_suffix, resolution, radius=None, n_neighbors=None):
    print("Creating gifti results masks")
    p_values_path = os.path.join(results_dir, f"p_values{hparam_suffix}.p")

    results_maps_path = os.path.join(results_dir, "results_maps")
    os.makedirs(results_maps_path, exist_ok=True)

    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    log_10_p_values = copy.deepcopy(p_values)
    log_10_p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    log_10_p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    for hemi in HEMIS:
        path_out = os.path.join(results_maps_path, f"p_values{hparam_suffix}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(log_10_p_values[hemi], path_out)

    # tfce values
    tfce_values_path = os.path.join(results_dir, f"tfce_values{hparam_suffix}.p")
    tfce_values = pickle.load(open(tfce_values_path, "rb"))

    for hemi in HEMIS:
        path_out = os.path.join(results_maps_path, f"tfce_values{hparam_suffix}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(tfce_values[hemi][metric], path_out)

    threshold = p_value_threshold
    if tfce_value_threshold is not None:
        threshold = tfce_value_threshold
        print(f"using tfce value threshold {tfce_value_threshold}")
        masks = {hemi: copy.deepcopy(tfce_values[hemi][metric]) for hemi in HEMIS}
        for hemi in HEMIS:
            masks[hemi][tfce_values[hemi][metric] > tfce_value_threshold] = 1
            masks[hemi][tfce_values[hemi][metric] <= tfce_value_threshold] = 0
            masks[hemi][np.isnan(tfce_values[hemi][metric])] = 0
            masks[hemi] = masks[hemi].astype(np.uint8)
    else:
        # p value masks
        masks = copy.deepcopy(p_values)
        for hemi in HEMIS:
            masks[hemi][p_values[hemi] < p_value_threshold] = 1
            masks[hemi][p_values[hemi] >= p_value_threshold] = 0
            masks[hemi][np.isnan(p_values[hemi])] = 0
            masks[hemi] = masks[hemi].astype(np.uint8)

    create_results_cluster_masks(masks, results_dir, hparam_suffix, metric, resolution, radius, n_neighbors, threshold)


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


def calc_tfce_values(t_values, edge_lengths_dicts, metric, h=2, e=1, dh=0.1, clip_value=100,
                     cluster_extents_measure="num_vertices"):
    tfce_values = dict()

    for hemi in HEMIS:
        values = t_values[hemi][metric]
        if len(values[values > clip_value]) > 0:
            print(f"Clipping {len(values[values > clip_value])} t-values that are greater than {clip_value}")
            values[values > clip_value] = clip_value

        max_score = np.nanmax(values)
        if np.isnan(max_score):
            print("encountered NaN in t-values while calculating tfce values")
            tfce_values[hemi] = {metric: np.zeros_like(values)}
            continue
        if np.isinf(max_score):
            print("encountered inf in t-values while calculating tfce values")
            tfce_values[hemi] = {metric: np.zeros_like(values)}
            continue

        if max_score <= 0:
            tfce_values[hemi] = {metric: np.zeros_like(values)}
            continue

        if dh == 0:
            dh = "auto"

        if dh == "auto":
            step = max_score / 100
            print(f"Automatically set dh to {step}")
        else:
            step = dh

        score_threshs = np.arange(step, max_score + step, step)

        tfce_values[hemi] = {metric: np.zeros(shape=values.shape, dtype=np.float32)}

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
