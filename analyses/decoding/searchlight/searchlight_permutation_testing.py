import argparse
import itertools
import math
import warnings

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

from analyses.cluster_analysis import get_edge_lengths_dicts_based_on_edges, calc_tfce_values, \
    calc_significance_cutoff, create_masks
from analyses.decoding.searchlight.searchlight import SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, \
    searchlight_mode_from_args, get_results_file_path
from data import MODALITY_AGNOSTIC, MODALITY_SPECIFIC_IMAGES, MODALITY_SPECIFIC_CAPTIONS, SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, LatentFeatsConfig, VISION_FEAT_COMBINATION_CHOICES, LANG_FEAT_COMBINATION_CHOICES, \
    TRAINING_MODES, SPLIT_TEST_IMAGES_ATTENDED, SPLIT_TEST_IMAGES_UNATTENDED, SPLIT_TEST_CAPTIONS_UNATTENDED, \
    SPLIT_TEST_CAPTIONS_ATTENDED, SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS, SPLIT_IMAGERY, SPLIT_IMAGERY_WEAK
from eval import CHANCE_VALUES, LIMITED_CANDIDATE_LATENTS
from utils import SUBJECTS_ADDITIONAL_TEST, HEMIS, DEFAULT_RESOLUTION, DATA_DIR, \
    METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC, \
    METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_DECODING, \
    DEFAULT_MODEL, METRIC_MOD_AGNOSTIC_AND_CROSS, FS_NUM_VERTICES, METRIC_DIFF_ATTENTION

DEFAULT_N_JOBS = 10

DIFF = "diff"

T_VAL_METRICS = [
    '$'.join([MODALITY_AGNOSTIC, SPLIT_IMAGERY]),
    '$'.join([MODALITY_AGNOSTIC, SPLIT_IMAGERY_WEAK]),
    '$'.join([MODALITY_AGNOSTIC, SPLIT_TEST_IMAGES]),
    '$'.join([MODALITY_AGNOSTIC, SPLIT_TEST_CAPTIONS]),
    '$'.join([MODALITY_SPECIFIC_IMAGES, SPLIT_TEST_CAPTIONS]),  # cross-modal decoding
    '$'.join([MODALITY_SPECIFIC_CAPTIONS, SPLIT_TEST_IMAGES]),  # cross-modal decoding
    # MODALITY_SPECIFIC_IMAGES, SPLIT_TEST_CAPTIONS_ATTENDED,  # cross-modal decoding
    # MODALITY_SPECIFIC_IMAGES, SPLIT_TEST_CAPTIONS_UNATTENDED,  # cross-modal decoding
    '$'.join([MODALITY_AGNOSTIC, SPLIT_TEST_IMAGES_ATTENDED]),
    '$'.join([MODALITY_AGNOSTIC, SPLIT_TEST_IMAGES_UNATTENDED]),
    '$'.join([MODALITY_AGNOSTIC, SPLIT_TEST_CAPTIONS_ATTENDED]),
    '$'.join([MODALITY_AGNOSTIC, SPLIT_TEST_CAPTIONS_UNATTENDED]),
    '$'.join([DIFF, MODALITY_AGNOSTIC, SPLIT_TEST_IMAGES_ATTENDED, SPLIT_TEST_IMAGES_UNATTENDED]),
    # TODO: mod-agnostic or specific decoder?
    '$'.join([DIFF, MODALITY_AGNOSTIC, SPLIT_TEST_CAPTIONS_ATTENDED, SPLIT_TEST_CAPTIONS_UNATTENDED]),
    # TODO: mod-agnostic or specific decoder?
    # DIFF+MODALITY_SPECIFIC_IMAGES+SPLIT_TEST_CAPTIONS_ATTENDED+SPLIT_TEST_CAPTIONS_UNATTENDED #TODO

    # DIFF+SPLIT_TEST_IMAGES+MODALITY_AGNOSTIC+MODALITY_SPECIFIC_IMAGES,
    # DIFF, SPLIT_TEST_CAPTIONS, MODALITY_AGNOSTIC, MODALITY_SPECIFIC_CAPTIONS,
]
TFCE_VAL_METRICS = [METRIC_CROSS_DECODING, METRIC_MOD_AGNOSTIC_AND_CROSS, METRIC_DIFF_ATTENTION]


# def add_diff_metrics(sc):
#     dfs_to_add = []
#     for training_mode in TRAINING_MODES:
#         for subject in tqdm(SUBJECTS_ADDITIONAL_TEST, desc=f'Adding {training_mode} decoder diff metrics'):
#             for hemi in HEMIS:
#                 attended = sc[(sc.training_mode == training_mode) & (sc.subject == subject) & (sc.hemi == hemi) & (
#                         sc.metric == SPLIT_TEST_IMAGES_ATTENDED)]
#                 unattended = sc[(sc.training_mode == training_mode) & (sc.subject == subject) & (sc.hemi == hemi) & (
#                         sc.metric == SPLIT_TEST_IMAGES_UNATTENDED)]
#                 assert len(attended) == len(unattended) == FS_NUM_VERTICES
#                 diff_imgs = attended.copy()
#                 diff_imgs['value'] = (attended.value.values - unattended.value.values)
#                 diff_imgs['metric'] = 'diff_attended_unattended_images'
#                 dfs_to_add.append(diff_imgs)
#
#                 attended = sc[(sc.training_mode == training_mode) & (sc.subject == subject) & (sc.hemi == hemi) & (
#                         sc.metric == SPLIT_TEST_CAPTIONS_ATTENDED)]
#                 unattended = sc[(sc.training_mode == training_mode) & (sc.subject == subject) & (sc.hemi == hemi) & (
#                         sc.metric == SPLIT_TEST_CAPTIONS_UNATTENDED)]
#                 assert len(attended) == len(unattended) == FS_NUM_VERTICES
#                 diff_caps = attended.copy()
#                 diff_caps['value'] = (attended.value.values - unattended.value.values)
#                 diff_caps['metric'] = 'diff_attended_unattended_captions'
#                 dfs_to_add.append(diff_caps)
#
#                 imgs = sc[(sc.training_mode == training_mode) & (sc.subject == subject) & (sc.hemi == hemi) & (
#                         sc.metric == SPLIT_TEST_IMAGES_ATTENDED)]
#                 caps = sc[(sc.training_mode == training_mode) & (sc.subject == subject) & (sc.hemi == hemi) & (
#                         sc.metric == SPLIT_TEST_CAPTIONS_ATTENDED)]
#                 assert len(imgs) == len(caps) == FS_NUM_VERTICES
#                 diff_attended = imgs.copy()
#                 diff_attended['value'] = (imgs.value.values - caps.value.values)
#                 diff_attended['metric'] = 'diff_images_captions_attended'
#                 dfs_to_add.append(diff_attended)
#
#                 imgs = sc[(sc.training_mode == training_mode) & (sc.subject == subject) & (sc.hemi == hemi) & (
#                         sc.metric == SPLIT_TEST_IMAGES_UNATTENDED)]
#                 caps = sc[(sc.training_mode == training_mode) & (sc.subject == subject) & (sc.hemi == hemi) & (
#                         sc.metric == SPLIT_TEST_CAPTIONS_UNATTENDED)]
#                 assert len(imgs) == len(caps) == FS_NUM_VERTICES
#                 diff_unattended = imgs.copy()
#                 diff_unattended['value'] = (imgs.value.values - caps.value.values)
#                 diff_unattended['metric'] = 'diff_images_captions_unattended'
#                 dfs_to_add.append(diff_unattended)
#
#     sc = pd.concat([sc] + dfs_to_add, ignore_index=True)
#
#     return sc


def load_per_subject_scores(args, hemis=HEMIS, latents=LIMITED_CANDIDATE_LATENTS, standardized_predictions='True'):
    print("loading per-subject scores")

    all_scores = []

    for subject in tqdm(args.subjects):
        for hemi in hemis:
            feats_config_mod_agnostic = LatentFeatsConfig(
                args.model,
                args.features,
                args.test_features,
                args.vision_features,
                args.lang_features,
            )
            results_mod_agnostic_file = get_results_file_path(
                feats_config_mod_agnostic, hemi, subject, MODALITY_AGNOSTIC,
                searchlight_mode_from_args(args), args.l2_regularization_alpha
            )
            scores_agnostic = pd.read_csv(results_mod_agnostic_file, index_col=0)
            # scores_agnostic['subject'] = subject #TODO temp
            # scores_agnostic['hemi'] = hemi  # TODO temp

            # scores_agnostic = results_agnostic['scores']
            # nan_locations = results_agnostic['nan_locations']
            # n_neighbors = results_agnostic['n_neighbors'] if 'n_neighbors' in results_agnostic else None
            # per_subject_n_neighbors[subject][hemi] = n_neighbors
            # per_subject_nan_locations[subject][hemi] = nan_locations

            feats_config_mod_specific_images = LatentFeatsConfig(
                args.mod_specific_images_model,
                args.mod_specific_images_features,
                args.mod_specific_images_test_features,
                args.vision_features,
                args.lang_features,
                logging=False
            )
            results_mod_specific_images_file = get_results_file_path(
                feats_config_mod_specific_images, hemi, subject, MODALITY_SPECIFIC_IMAGES,
                searchlight_mode_from_args(args), args.l2_regularization_alpha
            )
            if os.path.isfile(results_mod_specific_images_file):
                scores_images = pd.read_csv(results_mod_specific_images_file)
                # scores_images['subject'] = subject #TODO temp
                # scores_images['hemi'] = hemi #TODO temp
            else:
                print(f"Missing modality-specific results: {results_mod_specific_images_file}")
                scores_images = None

            feats_config_mod_specific_captions = LatentFeatsConfig(
                args.mod_specific_captions_model,
                args.mod_specific_captions_features,
                args.mod_specific_captions_test_features,
                args.vision_features,
                args.lang_features,
                logging=False
            )
            results_mod_specific_captions_file = get_results_file_path(
                feats_config_mod_specific_captions, hemi, subject, MODALITY_SPECIFIC_CAPTIONS,
                searchlight_mode_from_args(args), args.l2_regularization_alpha
            )
            if os.path.isfile(results_mod_specific_captions_file):
                scores_captions = pd.read_csv(results_mod_specific_captions_file)
                # scores_captions['subject'] = subject #TODO temp
                # scores_captions['hemi'] = hemi #TODO temp
            else:
                print(f"Missing modality-specific results: {results_mod_specific_captions_file}")
                scores_captions = pd.DataFrame()

            scores = pd.concat([scores_agnostic, scores_captions, scores_images], ignore_index=True)

            # print({n: round(np.nanmean(score), 4) for n, score in scores.items()})
            # print({f"{n}_max": round(np.nanmax(score), 2) for n, score in scores.items()})
            # print("")

            scores = scores[scores.latents == latents]
            scores = scores[scores.standardized_predictions == standardized_predictions]
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


def calc_t_values(per_subject_scores):
    t_values = {hemi: dict() for hemi in HEMIS}
    for hemi in HEMIS:
        for metric in T_VAL_METRICS:
            data = np.array([per_subject_scores[subj][hemi][metric] for subj in args.subjects])
            popmean = CHANCE_VALUES[metric]
            t_values[hemi][metric] = calc_image_t_values(data, popmean, use_tqdm=True, metric=metric)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            t_values[hemi][METRIC_MOD_AGNOSTIC_AND_CROSS] = np.nanmin(
                (
                    t_values[hemi][ACC_IMAGES_MOD_AGNOSTIC],
                    t_values[hemi][ACC_CAPTIONS_MOD_AGNOSTIC],
                    t_values[hemi][ACC_CAPTIONS_MOD_SPECIFIC_IMAGES],
                    t_values[hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS]),
                axis=0)
            t_values[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC] = np.nanmin(
                (
                    t_values[hemi][METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC],
                    t_values[hemi][METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC],
                    t_values[hemi][ACC_IMAGES_MOD_AGNOSTIC],
                    t_values[hemi][ACC_CAPTIONS_MOD_AGNOSTIC]),
                axis=0)
            t_values[hemi][METRIC_CROSS_DECODING] = np.nanmin(
                (
                    t_values[hemi][ACC_CAPTIONS_MOD_SPECIFIC_IMAGES],
                    t_values[hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS]
                ),
                axis=0
            )

    return t_values


def calc_test_statistics(args):
    t_values_path = os.path.join(permutation_results_dir(args), "t_values.p")
    if not os.path.isfile(t_values_path):
        print(f"Calculating t-values")
        per_subject_scores = load_per_subject_scores(args)
        t_values = calc_t_values(per_subject_scores)
        pickle.dump(t_values, open(t_values_path, 'wb'))
    else:
        t_values = pickle.load(open(t_values_path, 'rb'))

    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
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
        f"tfce_values_null_distribution{get_hparam_suffix(args)}.p"
    )
    null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))
    significance_cutoff, max_test_statistic_distr = calc_significance_cutoff(null_distribution_tfce_values, args.metric,
                                                                             args.p_value_threshold)

    p_values = {hemi: np.repeat(np.nan, t_values[hemi][args.metric].shape) for hemi, t_vals in t_values.items()}
    for hemi in HEMIS:
        print(f"{hemi} hemi largest test statistic values: ",
              sorted([t for t in tfce_values[hemi][args.metric]], reverse=True)[:10])
        print(f"{hemi} hemi largest test statistic null distr values: ", max_test_statistic_distr[-10:])
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


# def assemble_null_distr_per_subject_scores(subject, args):
#     print(f"assembling {subject} null distr scores")
#     null_distr_dir = os.path.join(permutation_results_dir(args), 'null_distr_assembled')
#     os.makedirs(null_distr_dir, exist_ok=True)
#
#     for hemi in HEMIS:
#         def load_null_distr_scores(base_path):
#             scores_dir = os.path.join(os.path.dirname(base_path), "null_distr")
#             print(f'loading scores from {scores_dir}')
#             score_paths = sorted(list(glob(os.path.join(scores_dir, "*.p"))))
#             if len(score_paths) == 0:
#                 raise RuntimeError(f"No null distribution scores found: {scores_dir}")
#             last_idx = int(os.path.basename(score_paths[-1])[:-2])
#             assert last_idx == len(score_paths) - 1, f"{last_idx} vs. {len(score_paths)}"
#
#             def load_scores_from_pickle(paths, proc_id):
#                 job_scores = []
#                 iterator = tqdm(paths) if proc_id == args.n_jobs-1 else paths
#                 for path in iterator:
#                     scores = pickle.load(open(path, "rb"))
#                     for scores_perm in scores:
#                         scores_perm['vertex'] = int(os.path.basename(path)[:-2])
#                         # scores_perm['subject'] = subject #not necessary
#                         # scores_perm['hemi'] = hemi #not necessary
#                     job_scores.append(scores)
#                 return job_scores
#
#             n_per_job = math.ceil(len(score_paths) / args.n_jobs)
#             all_scores = Parallel(n_jobs=args.n_jobs)(
#                 delayed(load_scores_from_pickle)(
#                     score_paths[id * n_per_job:(id + 1) * n_per_job],
#                     id,
#                 )
#                 for id in range(args.n_jobs)
#             )
#
#             return np.concatenate(all_scores)
#
#         for training_mode in TRAINING_MODES:
#             if training_mode == MODALITY_AGNOSTIC:
#                 feats_config = LatentFeatsConfig(
#                     args.model,
#                     args.features,
#                     args.test_features,
#                     args.vision_features,
#                     args.lang_features,
#                     logging=False
#                 )
#             elif training_mode == MODALITY_SPECIFIC_IMAGES:
#                 feats_config = LatentFeatsConfig(
#                     args.mod_specific_images_model,
#                     args.mod_specific_images_features,
#                     args.mod_specific_images_test_features,
#                     args.vision_features,
#                     args.lang_features,
#                     logging=False
#                 )
#             elif training_mode == MODALITY_SPECIFIC_CAPTIONS:
#                 feats_config = LatentFeatsConfig(
#                     args.mod_specific_captions_model,
#                     args.mod_specific_captions_features,
#                     args.mod_specific_captions_test_features,
#                     args.vision_features,
#                     args.lang_features,
#                     logging=False
#                 )
#             else:
#                 raise RuntimeError(f"Unknown training mode: {training_mode}")
#
#             results_file = get_results_file_path(
#                 feats_config, hemi, subject, training_mode,
#                 searchlight_mode_from_args(args), args.l2_regularization_alpha,
#             )
#             null_distribution = load_null_distr_scores(results_file)
#
#             num_permutations = len(null_distribution[0])
#             print(f'final per subject scores null distribution dict creation for {training_mode} decoder:')
#             for perm_id in tqdm(range(num_permutations)):
#                 scores = pd.concat([null_distr[perm_id] for null_distr in null_distribution], ignore_index=True)
#                 # distr_caps = pd.concat([null_distr[perm_id] for null_distr in null_distribution_captions],
#                 #                        ignore_index=True)
#                 # distr_imgs = pd.concat([null_distr[perm_id] for null_distr in null_distribution_images], ignore_index=True)
#                 # scores = pd.concat([distr, distr_caps, distr_imgs], ignore_index=True)
#                 subject_scores_null_distr_path = os.path.join(null_distr_dir,
#                                                               f"{subject}_scores_null_distr_{training_mode}_{hemi}_hemi_{perm_id}.p")
#                 pickle.dump(scores, open(subject_scores_null_distr_path, 'wb'))


def calc_t_values_null_distr(args, out_path):
    # per_subject_scores_null_distr = dict()
    # for subject in tqdm(args.subjects):
    #     subject_scores_null_distr_dir = os.path.join(permutation_results_dir(args), f"null_distr_assembled")
    # if not os.path.isdir(subject_scores_null_distr_dir):
    #     assemble_null_distr_per_subject_scores(subject, args)
    # else:
    #     print(f"loading assembled null distr scores for {subject}")
    #     per_subject_scores_null_distr[subject] = pickle.load(open(subject_scores_null_distr_path, 'rb'))

    # subject_scores_null_distr_path = os.path.join(permutation_results_dir(args), 'null_distr_assembled',
    #                                               f"{args.subjects[0]}_scores_null_distr_{HEMIS[0]}_hemi_0.p")
    # sample_null_distr = pickle.load(open(subject_scores_null_distr_path, 'rb'))

    # n_permutations = len(glob(os.path.join(subject_scores_null_distr_dir,
    #                                               f"{args.subjects[0]}_scores_null_distr_{MODALITY_AGNOSTIC}_{HEMIS[0]}_hemi_**.p")))

    def calc_permutation_t_values(vertex_range, permutations, proc_id, tmp_file_path, subjects,
                                  latents_mode='limited_candidate_latents', standardized_predictions=True):
        os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)

        with h5py.File(tmp_file_path, 'w') as f:
            dsets = dict()
            for hemi in HEMIS:
                dsets[hemi] = dict()
                for metric in T_VAL_METRICS + TFCE_VAL_METRICS:
                    tvals_shape = (len(permutations), vertex_range[1] - vertex_range[0])
                    dsets[hemi][metric] = f.create_dataset(f"{hemi}__{metric}", tvals_shape, dtype='float16')

            if proc_id == args.n_jobs - 1:
                permutations_iterator = tqdm(enumerate(permutations), total=len(permutations),
                                             desc="calculating null distr t-vals")
                print('preloading null distr scores')
            else:
                permutations_iterator = enumerate(permutations)

            preloaded_scores = dict()
            for subj in subjects:
                preloaded_scores[subj] = dict()
                for hemi in HEMIS:
                    preloaded_scores[subj][hemi] = dict()
                    for training_mode in TRAINING_MODES:
                        preloaded_scores[subj][hemi][training_mode] = dict()

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
                        preloaded_scores[subj][hemi][training_mode] = []

                        gathered_over_vertices = dict()

                        for vertex_id in vertex_iter:
                            gathered_over_vertices[vertex_id] = []
                            scores_path = os.path.join(os.path.dirname(base_path), "null_distr",
                                                       f"{vertex_id:010d}.p")
                            scores_vertex = pickle.load(open(scores_path, "rb"))
                            # scores_vertex['vertex'] = vertex_id
                            # scores_vertex['training_mode'] = training_mode
                            # null_distr_scores.append(scores_vertex)
                            # preloaded_scores[subj][hemi][training_mode][vertex_id] = []
                            for scores_perm in scores_vertex:
                                scores_perm = scores_perm[(scores_perm.latents == latents_mode) & (
                                        scores_perm.standardized_predictions == standardized_predictions)]
                                metrics = scores_perm[['metric', 'value']].set_index('metric').value.to_dict()

                                gathered_over_vertices[vertex_id].append(metrics)
                                # saving in format [subj][hemi][training_mode][perm_id][metric]
                        for perm_id in range(len(gathered_over_vertices[vertex_range[0]])):
                            gathered = {metric: np.array([gathered_over_vertices[i][perm_id][metric] for i in range(vertex_range[0], vertex_range[1])]) for metric in gathered_over_vertices[vertex_range[0]][perm_id].keys()}
                            preloaded_scores[subj][hemi][training_mode].append(gathered)

                    # null_distr_scores = pd.concat(null_distr_scores, ignore_index=True)

            for iteration, permutation in permutations_iterator:
                t_values = {hemi: dict() for hemi in HEMIS}
                for hemi in HEMIS:
                    for metric in T_VAL_METRICS:
                        data = np.zeros((len(args.subjects), vertex_range[1] - vertex_range[0]))
                        for i, (idx, subj) in enumerate(zip(permutation, args.subjects)):
                            if metric.startswith(DIFF):
                                training_mode, metric_name_1, metric_name_2 = metric.split('$')[1:]
                                data[i] = preloaded_scores[subj][hemi][training_mode][idx][metric_name_1] - \
                                          preloaded_scores[subj][hemi][training_mode][idx][metric_name_2]
                            else:
                                training_mode, metric_name = metric.split('$')
                                data[i] = preloaded_scores[subj][hemi][training_mode][idx][metric_name]

                        popmean = 0 if metric.startswith(DIFF) else 0.5
                        t_values[hemi][metric] = calc_image_t_values(data, popmean)
                        dsets[hemi][metric][iteration] = t_values[hemi][metric].astype(np.float16)
                        #
                        # for training_mode in TRAINING_MODES:
                        #     t_values[hemi][training_mode] = dict()
                        #     for metric in T_VAL_METRICS:
                        #         # data = np.array(
                        #         #     [assembled[i][assembled[i].metric == metric].value.values for i in range(len(assembled))])
                        #         # print(data.shape)
                        #         # print(data)
                        #         data = np.array(
                        #             [preloaded_scores[subj][hemi][vertex_id][idx][metric] for idx, subj in
                        #              zip(permutation, args.subjects)])
                        #         # data = np.array(
                        #     [per_subject_scores[idx][subj][hemi][metric] for idx, subj in
                        #      zip(permutation, args.subjects)])
                        # popmean = CHANCE_VALUES[metric]
                        # t_values[hemi][training_mode][metric] = calc_image_t_values(data, popmean).astype(
                        #     np.float16)
                        # dsets[hemi][training_mode][metric][iteration] = t_values[hemi][metric]


                    #TODO: revise complex metrics
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        dsets[hemi][METRIC_MOD_AGNOSTIC_AND_CROSS][iteration] = np.nanmin(
                            (
                                t_values[hemi]['$'.join([MODALITY_AGNOSTIC, SPLIT_TEST_IMAGES])],
                                t_values[hemi]['$'.join([MODALITY_AGNOSTIC, SPLIT_TEST_CAPTIONS])],
                                t_values[hemi]['$'.join([MODALITY_SPECIFIC_IMAGES, SPLIT_TEST_CAPTIONS])],
                                t_values[hemi]['$'.join([MODALITY_SPECIFIC_CAPTIONS, SPLIT_TEST_IMAGES])]),
                            axis=0)
                        dsets[hemi][METRIC_DIFF_ATTENTION][iteration] = np.nanmin(
                            (
                                 t_values[hemi]['$'.join([DIFF, MODALITY_AGNOSTIC, SPLIT_TEST_IMAGES_ATTENDED, SPLIT_TEST_IMAGES_UNATTENDED])],
                                 t_values[hemi]['$'.join([DIFF, MODALITY_AGNOSTIC, SPLIT_TEST_CAPTIONS_ATTENDED,SPLIT_TEST_CAPTIONS_UNATTENDED])]),
                            axis=0)


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

    # n_vertices = {
    #     hemi: pickle.load(open(os.path.join(subject_scores_null_distr_dir, f"{args.subjects[0]}_scores_null_distr_{MODALITY_AGNOSTIC}_{hemi}_hemi_0.p"), 'rb'))['vertex'].max()+1 for hemi in
    #     HEMIS
    # }

    # scores_jobs = {job_id: [] for job_id in range(args.n_jobs)}

    vertex_ranges = [(job_id * n_per_job, min((job_id + 1) * n_per_job, n_vertices)) for job_id in range(args.n_jobs)]
    print('vertex ranges for jobs: ', vertex_ranges)

    # for perm_id in trange(n_permutations, desc="splitting up for jobs"):
    #     for job_id in range(args.n_jobs):
    #         scores_jobs[job_id].append({s: {hemi: dict() for hemi in HEMIS} for s in args.subjects})
    #     for subj in args.subjects:
    #         for hemi in HEMIS:
    #             subject_scores_null_distr_path = os.path.join(subject_scores_null_distr_dir,
    #                                                           f"{subj}_scores_null_distr_{training_mode}_{hemi}_hemi_{perm_id}.p")
    #             sample_null_distr = pickle.load(open(subject_scores_null_distr_path, 'rb'))
    #             for metric in sample_null_distr.metric.unique(): #TODO load data within job only!
    #                 for job_id in range(args.n_jobs):
    #                     # TODO
    #                     scores_job = per_subject_scores_null_distr[subj][perm_id][hemi][metric]
    #                     filtered = scores_job[job_id * n_per_job[hemi]:(job_id + 1) * n_per_job[hemi]]
    #                     scores_jobs[job_id][perm_id][subj][hemi][metric] = filtered

    tmp_filenames = {job_id: os.path.join(os.path.dirname(out_path), f"temp_t_vals", f"{job_id}.hdf5") for job_id in
                     range(args.n_jobs)}

    # # TODO single iter for debugging
    # calc_permutation_t_values(vertex_ranges[-1], permutations, args.n_jobs - 1, tmp_filenames[args.n_jobs - 1],
    #                           args.subjects)
    Parallel(n_jobs=args.n_jobs, mmap_mode=None, max_nbytes=None)(
        delayed(calc_permutation_t_values)(
            vertex_ranges[id],
            permutations,
            id,
            tmp_filenames[id],
            args.subjects,
        )
        for id in range(args.n_jobs)
    )
    print('finished calculating null distr t-vals')

    tmp_files = dict()
    for job_id in range(args.n_jobs):
        tmp_files[job_id] = h5py.File(tmp_filenames[job_id], 'r')

    with h5py.File(out_path, 'w') as all_t_vals_file:
        for hemi_metric in tmp_files[0].keys():
            # hemi = hemi_metric.split('__')[0]
            tvals_shape = (args.n_permutations_group_level, n_vertices)
            all_t_vals_file.create_dataset(hemi_metric, tvals_shape, dtype='float32', fillvalue=np.nan)

        for i in tqdm(range(args.n_permutations_group_level), desc="assembling results"):
            for hemi_metric in tmp_files[0].keys():
                data_tvals = np.concatenate([tmp_files[job_id][hemi_metric][i] for job_id in range(args.n_jobs)])
                all_t_vals_file[hemi_metric][i] = data_tvals

    print("finished assemble")


def permutation_results_dir(args):
    return str(os.path.join(
        SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, args.model, args.features, args.mod_specific_images_model,
        args.mod_specific_images_features, args.mod_specific_captions_model, args.mod_specific_captions_features,
        args.resolution, searchlight_mode_from_args(args)
    ))


def get_hparam_suffix(args):
    return f"_{args.metric}_h_{args.tfce_h}_e_{args.tfce_e}_dh_{args.tfce_dh}"


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
                indices = range(proc_id * n_per_job, min((proc_id + 1) * n_per_job, args.n_permutations_group_level))
                iterator = tqdm(indices) if proc_id == 0 else indices
                tfce_values = []
                for iteration in iterator:
                    vals = {hemi: {args.metric: t_vals[f"{hemi}__{args.metric}"][iteration]} for hemi in HEMIS}
                    tfce_values.append(
                        calc_tfce_values(
                            vals, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e, dh=args.tfce_dh,
                        )
                    )
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


def add_searchlight_permutation_args(parser):
    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS_ADDITIONAL_TEST)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--mod-specific-images-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-images-features", type=str, default=SELECT_DEFAULT)
    parser.add_argument("--mod-specific-images-test-features", type=str, default=SELECT_DEFAULT)

    parser.add_argument("--mod-specific-captions-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-captions-features", type=str, default=SELECT_DEFAULT)
    parser.add_argument("--mod-specific-captions-test-features", type=str, default=SELECT_DEFAULT)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)
    parser.add_argument("--tfce-dh", type=float, default=0.1)

    parser.add_argument("--metric", type=str, default=METRIC_MOD_AGNOSTIC_AND_CROSS)

    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-permutations-group-level", type=int, default=1000)

    parser.add_argument("--p-value-threshold", type=float, default=1e-4)
    parser.add_argument("--tfce-value-threshold", type=float, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, exist_ok=True)
    args = get_args()

    print(f"\n\nPermutation Testing for {args.metric}\n")
    create_null_distribution(args)
    calc_test_statistics(args)
    create_masks(permutation_results_dir(args), args.metric, args.p_value_threshold, args.tfce_value_threshold,
                 get_hparam_suffix(args),
                 args.resolution, args.radius, args.n_neighbors)
