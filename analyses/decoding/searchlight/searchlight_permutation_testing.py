import argparse
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
from scipy.spatial.distance import cdist
from tqdm import tqdm, trange

from analyses.cluster_analysis import get_edge_lengths_dicts_based_on_edges, calc_tfce_values, \
    calc_significance_cutoff, create_masks
from analyses.decoding.ridge_regression_decoding import ACC_CAPTIONS, ACC_IMAGES
from analyses.decoding.searchlight.searchlight import SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, \
    searchlight_mode_from_args, get_results_file_path
from data import MODALITY_AGNOSTIC, MODALITY_SPECIFIC_IMAGES, MODALITY_SPECIFIC_CAPTIONS, SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, LatentFeatsConfig, VISION_FEAT_COMBINATION_CHOICES, LANG_FEAT_COMBINATION_CHOICES
from eval import ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, ACC_IMAGES_MOD_SPECIFIC_IMAGES, \
    ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS, ACC_CAPTIONS_MOD_AGNOSTIC, \
    ACC_IMAGERY_MOD_AGNOSTIC, ACC_IMAGES_MOD_AGNOSTIC, ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, \
    ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGERY_MOD_SPECIFIC_CAPTIONS, \
    ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_NO_STD_MOD_SPECIFIC_CAPTIONS, \
    ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGERY_NO_STD_MOD_SPECIFIC_IMAGES, ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGERY_MOD_SPECIFIC_IMAGES, CHANCE_VALUES
from utils import SUBJECTS, HEMIS, DEFAULT_RESOLUTION, DATA_DIR, METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC, \
    METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_DECODING, \
    DEFAULT_MODEL, METRIC_MOD_AGNOSTIC_AND_CROSS

DEFAULT_N_JOBS = 10

T_VAL_METRICS = [
    METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC, METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC,
    ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS,
    ACC_IMAGES_MOD_AGNOSTIC, ACC_CAPTIONS_MOD_AGNOSTIC, ACC_IMAGERY_MOD_AGNOSTIC,
    ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES,
    ACC_IMAGES_MOD_SPECIFIC_CAPTIONS
]

MIN_NUM_DATAPOINTS = 4


def process_scores(scores_agnostic, scores_mod_specific_captions, scores_mod_specific_images, nan_locations,
                   additional_imagery_scores=False):
    scores = dict()

    metrics = [ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST]
    metric_names = [ACC_CAPTIONS_MOD_AGNOSTIC, ACC_IMAGES_MOD_AGNOSTIC, ACC_IMAGERY_MOD_AGNOSTIC,
                    ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC]
    # if additional_imagery_scores:
    #     metrics += [ACC_IMAGERY + "_no_std", ACC_IMAGERY_WHOLE_TEST + "_no_std"]
    #     metric_names += [ACC_IMAGERY_NO_STD_MOD_AGNOSTIC, ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_AGNOSTIC]
    for metric_agnostic_name, metric in zip(metric_names, metrics):
        scores[metric_agnostic_name] = np.repeat(np.nan, nan_locations.shape)
        scores[metric_agnostic_name][~nan_locations] = np.array([score[metric] for score in scores_agnostic])

    if scores_mod_specific_captions is not None and scores_mod_specific_images is not None:
        metric_names = [ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS, ACC_IMAGES_MOD_SPECIFIC_CAPTIONS]
        metrics = [ACC_CAPTIONS, ACC_IMAGES]
        if additional_imagery_scores:
            metrics += [ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, ACC_IMAGERY + "_no_std",
                        ACC_IMAGERY_WHOLE_TEST + "_no_std"]
            metric_names += [ACC_IMAGERY_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_CAPTIONS,
                             ACC_IMAGERY_NO_STD_MOD_SPECIFIC_CAPTIONS,
                             ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_CAPTIONS]
        for metric_specific_name, metric in zip(metric_names, metrics):
            scores[metric_specific_name] = np.repeat(np.nan, nan_locations.shape)
            scores[metric_specific_name][~nan_locations] = np.array(
                [score[metric] for score in scores_mod_specific_captions])

        metric_names = [ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES]
        metrics = [ACC_IMAGES, ACC_CAPTIONS]
        if additional_imagery_scores:
            metrics += [ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, ACC_IMAGERY + "_no_std",
                        ACC_IMAGERY_WHOLE_TEST + "_no_std"]
            metric_names += [ACC_IMAGERY_MOD_SPECIFIC_IMAGES, ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_IMAGES,
                             ACC_IMAGERY_NO_STD_MOD_SPECIFIC_IMAGES,
                             ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_IMAGES]
        for metric_specific_name, metric in zip(metric_names, metrics):
            scores[metric_specific_name] = np.repeat(np.nan, nan_locations.shape)
            scores[metric_specific_name][~nan_locations] = np.array(
                [score[metric] for score in scores_mod_specific_images])

        scores[METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC] = np.array(
            [ai - si for ai, si in zip(scores[ACC_IMAGES_MOD_AGNOSTIC], scores[ACC_IMAGES_MOD_SPECIFIC_IMAGES])]
        )
        scores[METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC] = np.array(
            [ac - sc for ac, sc in zip(scores[ACC_CAPTIONS_MOD_AGNOSTIC], scores[ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS])]
        )

    return scores


def load_per_subject_scores(args, return_nan_locations_and_n_neighbors=False, hemis=HEMIS,
                            additional_imagery_scores=False):
    print("loading per-subject scores")

    per_subject_scores = {subj: dict() for subj in args.subjects}
    per_subject_n_neighbors = {subj: dict() for subj in args.subjects}
    per_subject_nan_locations = {subj: dict() for subj in args.subjects}

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
                feats_config_mod_agnostic, hemi, subject, MODALITY_AGNOSTIC, args.resolution,
                searchlight_mode_from_args(args), args.l2_regularization_alpha
            )
            results_agnostic = pickle.load(open(results_mod_agnostic_file, 'rb'))
            scores_agnostic = results_agnostic['scores']
            nan_locations = results_agnostic['nan_locations']
            n_neighbors = results_agnostic['n_neighbors'] if 'n_neighbors' in results_agnostic else None
            per_subject_n_neighbors[subject][hemi] = n_neighbors
            per_subject_nan_locations[subject][hemi] = nan_locations

            feats_config_mod_specific_images = LatentFeatsConfig(
                args.mod_specific_images_model,
                args.mod_specific_images_features,
                args.mod_specific_images_test_features,
                args.vision_features,
                args.lang_features,
                logging=False
            )
            results_mod_specific_images_file = get_results_file_path(
                feats_config_mod_specific_images, hemi, subject, MODALITY_SPECIFIC_IMAGES, args.resolution,
                searchlight_mode_from_args(args), args.l2_regularization_alpha
            )
            if os.path.isfile(results_mod_specific_images_file):
                scores_images = pickle.load(open(results_mod_specific_images_file, 'rb'))['scores']
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
                feats_config_mod_specific_captions, hemi, subject, MODALITY_SPECIFIC_CAPTIONS, args.resolution,
                searchlight_mode_from_args(args), args.l2_regularization_alpha
            )
            if os.path.isfile(results_mod_specific_captions_file):
                scores_captions = pickle.load(open(results_mod_specific_captions_file, 'rb'))['scores']
            else:
                print(f"Missing modality-specific results: {results_mod_specific_captions_file}")
                scores_captions = None

            scores = process_scores(scores_agnostic, scores_captions, scores_images, nan_locations,
                                    additional_imagery_scores)

            # print({n: round(np.nanmean(score), 4) for n, score in scores.items()})
            # print({f"{n}_max": round(np.nanmax(score), 2) for n, score in scores.items()})
            # print("")

            per_subject_scores[subject][hemi] = scores

    if return_nan_locations_and_n_neighbors:
        return per_subject_scores, per_subject_nan_locations, per_subject_n_neighbors
    else:
        return per_subject_scores


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
    values_no_nan = values[~np.isnan(values)]
    # use heuristic (mean needs to be greater than popmean) to speed up calculation
    # if values_no_nan.mean() > popmean:
    if np.all(values_no_nan.round(2) == values_no_nan[0].round(2)):
        # If all values are (almost) equal, the t-value would be disproportionally high, so we discard the value
        t_val = np.nan
    else:
        t_val = ttest_1samp_no_p(values_no_nan-popmean, sigma=sigma)
        if t_val > 50:
            print(f't val {t_val} for values {values_no_nan}')
    return t_val
    # else:
    #     return 0


def calc_image_t_values(data, popmean, use_tqdm=False, precision=None, metric=None, sigma=0):
    if precision is not None:
        data = data.round(precision)

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
            enough_data = np.argwhere(((~np.isnan(data)).sum(axis=0)) >= MIN_NUM_DATAPOINTS)[:, 0]
            t_values[hemi][metric] = np.repeat(np.nan, data.shape[1])
            t_values[hemi][metric][enough_data] = calc_image_t_values(
                data[:, enough_data], popmean, use_tqdm=True, metric=metric
            )

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
                    t_values[hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS],
                    t_values[hemi][ACC_IMAGES_MOD_SPECIFIC_IMAGES],
                    t_values[hemi][ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS]),
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


def assemble_null_distr_per_subject_scores(subject, args):
    print(f"assembling {subject} null distr scores")

    subject_scores_null_distr = []
    for hemi in HEMIS:
        feats_config_mod_agnostic = LatentFeatsConfig(
            args.model,
            args.features,
            args.test_features,
            args.vision_features,
            args.lang_features,
            logging=False
        )
        results_mod_agnostic_file = get_results_file_path(
            feats_config_mod_agnostic, hemi, subject, MODALITY_AGNOSTIC, args.resolution,
            searchlight_mode_from_args(args), args.l2_regularization_alpha,
        )
        results_agnostic = pickle.load(open(results_mod_agnostic_file, 'rb'))
        nan_locations = results_agnostic['nan_locations']

        feats_config_mod_specific_images = LatentFeatsConfig(
            args.mod_specific_images_model,
            args.mod_specific_images_features,
            args.mod_specific_images_test_features,
            args.vision_features,
            args.lang_features,
            logging=False
        )
        results_mod_specific_images_file = get_results_file_path(
            feats_config_mod_specific_images, hemi, subject, MODALITY_SPECIFIC_IMAGES, args.resolution,
            searchlight_mode_from_args(args), args.l2_regularization_alpha,
        )

        feats_config_mod_specific_captions = LatentFeatsConfig(
            args.mod_specific_captions_model,
            args.mod_specific_captions_features,
            args.mod_specific_captions_test_features,
            args.vision_features,
            args.lang_features,
            logging=False
        )
        results_mod_specific_captions_file = get_results_file_path(
            feats_config_mod_specific_captions, hemi, subject, MODALITY_SPECIFIC_CAPTIONS, args.resolution,
            searchlight_mode_from_args(args), f'alpha_{str(args.l2_regularization_alpha)}.p'
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

        null_distribution_agnostic = load_null_distr_scores(os.path.dirname(results_mod_agnostic_file))
        null_distribution_images = load_null_distr_scores(os.path.dirname(results_mod_specific_images_file))
        null_distribution_captions = load_null_distr_scores(os.path.dirname(results_mod_specific_captions_file))

        num_permutations = len(null_distribution_agnostic[0])
        print('final per subject scores null distribution dict creation:')
        for i in tqdm(range(num_permutations)):
            distr = [null_distr[i] for null_distr in null_distribution_agnostic]
            distr_caps = [null_distr[i] for null_distr in null_distribution_captions]
            distr_imgs = [null_distr[i] for null_distr in null_distribution_images]
            if len(subject_scores_null_distr) <= i:
                subject_scores_null_distr.append(dict())
            scores = process_scores(distr, distr_caps, distr_imgs, nan_locations)
            subject_scores_null_distr[i][hemi] = scores

    subject_scores_null_distr_path = os.path.join(
        permutation_results_dir(args), f"{subject}_scores_null_distr.p"
    )
    pickle.dump(subject_scores_null_distr, open(subject_scores_null_distr_path, 'wb'))
    return subject_scores_null_distr


def calc_t_values_null_distr(args, out_path):
    per_subject_scores_null_distr = dict()
    for subject in tqdm(args.subjects):
        subject_scores_null_distr_path = os.path.join(
            permutation_results_dir(args), f"{subject}_scores_null_distr.p"
        )
        if not os.path.isfile(subject_scores_null_distr_path):
            per_subject_scores_null_distr[subject] = assemble_null_distr_per_subject_scores(subject, args)
        else:
            print(f"loading assembled null distr scores for {subject}")
            per_subject_scores_null_distr[subject] = pickle.load(open(subject_scores_null_distr_path, 'rb'))

    def calc_permutation_t_values(per_subject_scores, permutations, proc_id, tmp_file_path, subjects):
        os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)

        with h5py.File(tmp_file_path, 'w') as f:
            dsets = dict()
            for hemi in HEMIS:
                dsets[hemi] = dict()
                for metric in T_VAL_METRICS + [METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_DECODING,
                                               METRIC_MOD_AGNOSTIC_AND_CROSS]:
                    tvals_shape = (
                        len(permutations), per_subject_scores[0][subjects[0]][hemi][ACC_IMAGES_MOD_AGNOSTIC].size)
                    dsets[hemi][metric] = f.create_dataset(f"{hemi}__{metric}", tvals_shape, dtype='float32')

            if proc_id == 0:
                iterator = tqdm(enumerate(permutations), total=len(permutations), desc="calculating null distr t-vals")
            else:
                iterator = enumerate(permutations)

            for iteration, permutation in iterator:
                t_values = {hemi: dict() for hemi in HEMIS}
                for hemi in HEMIS:
                    for metric in T_VAL_METRICS:
                        data = np.array(
                            [per_subject_scores[idx][subj][hemi][metric] for idx, subj in
                             zip(permutation, args.subjects)])
                        popmean = CHANCE_VALUES[metric]
                        t_values[hemi][metric] = calc_image_t_values(data, popmean)
                        dsets[hemi][metric][iteration] = t_values[hemi][metric]

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        dsets[hemi][METRIC_MOD_AGNOSTIC_AND_CROSS][iteration] = np.nanmin(
                            (
                                t_values[hemi][ACC_IMAGES_MOD_AGNOSTIC],
                                t_values[hemi][ACC_CAPTIONS_MOD_AGNOSTIC],
                                t_values[hemi][ACC_CAPTIONS_MOD_SPECIFIC_IMAGES],
                                t_values[hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS]),
                            axis=0)
                        dsets[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC][iteration] = np.nanmin(
                            (
                                t_values[hemi][METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC],
                                t_values[hemi][METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC],
                                t_values[hemi][ACC_IMAGES_MOD_AGNOSTIC],
                                t_values[hemi][ACC_CAPTIONS_MOD_AGNOSTIC]),
                            axis=0)
                        dsets[hemi][METRIC_CROSS_DECODING][iteration] = np.nanmin(
                            (
                                t_values[hemi][ACC_IMAGES_MOD_SPECIFIC_CAPTIONS],
                                t_values[hemi][ACC_CAPTIONS_MOD_SPECIFIC_IMAGES],
                                t_values[hemi][ACC_IMAGES_MOD_SPECIFIC_IMAGES],
                                t_values[hemi][ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS]),
                            axis=0
                        )

    permutations_iter = itertools.permutations(range(len(per_subject_scores_null_distr[args.subjects[0]])), len(args.subjects))
    permutations = [next(permutations_iter) for _ in range(args.n_permutations_group_level)]

    n_vertices = per_subject_scores_null_distr[args.subjects[0]][0][HEMIS[0]][ACC_IMAGES_MOD_AGNOSTIC].shape[0]
    enough_data = {
        hemi: np.argwhere(
            (~np.isnan(
                [per_subject_scores_null_distr[subj][0][hemi][ACC_IMAGES_MOD_AGNOSTIC] for subj in args.subjects])).sum(
                axis=0) >= MIN_NUM_DATAPOINTS)[:, 0]
        for hemi in HEMIS
    }
    enough_data_lengths = {hemi: len(e) for hemi, e in enough_data.items()}
    print(f"original n vertices: {n_vertices} | enough data: {enough_data_lengths}")

    n_per_job = {hemi: math.ceil(len(enough_data[hemi]) / args.n_jobs) for hemi in HEMIS}
    print(f"n vertices per job: {n_per_job}")

    scores_jobs = {job_id: [] for job_id in range(args.n_jobs)}
    desc = "filtering scores for enough data and splitting up for jobs"
    for perm_id in trange(len(per_subject_scores_null_distr[args.subjects[0]]), desc=desc):
    # for id, scores in tqdm(enumerate(per_subject_scores_null_distr), total=len(per_subject_scores_null_distr[0]),
    #                        desc=desc):
        for job_id in range(args.n_jobs):
            scores_jobs[job_id].append({s: {hemi: dict() for hemi in HEMIS} for s in args.subjects})
        for subj in args.subjects:
            for hemi in HEMIS:
                for metric in per_subject_scores_null_distr[subj][perm_id][hemi].keys():
                    for job_id in range(args.n_jobs):
                        filtered = per_subject_scores_null_distr[subj][perm_id][hemi][metric][enough_data[hemi]]
                        scores_jobs[job_id][perm_id][subj][hemi][metric] = filtered[
                                                                      job_id * n_per_job[hemi]:(job_id + 1) * n_per_job[
                                                                          hemi]]
                # for metric in scores[subj][hemi].keys():
                #     for job_id in range(args.n_jobs):
                #         filtered = scores[subj][hemi][metric][enough_data[hemi]]
                #         scores_jobs[job_id][id][subj][hemi][metric] = filtered[
                #                                                       job_id * n_per_job[hemi]:(job_id + 1) * n_per_job[
                #                                                           hemi]]

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

    tmp_files = dict()
    for job_id in range(args.n_jobs):
        tmp_files[job_id] = h5py.File(tmp_filenames[job_id], 'r')

    with h5py.File(out_path, 'w') as all_t_vals_file:
        for hemi_metric in tmp_files[0].keys():
            tvals_shape = (args.n_permutations_group_level, n_vertices)
            all_t_vals_file.create_dataset(hemi_metric, tvals_shape, dtype='float32', fillvalue=np.nan)

        for i in tqdm(range(args.n_permutations_group_level), desc="assembling results"):
            for hemi_metric in tmp_files[0].keys():
                hemi = hemi_metric.split('__')[0]
                data_tvals = np.repeat(np.nan, n_vertices)
                data_tvals[enough_data[hemi]] = np.concatenate(
                    [tmp_files[job_id][hemi_metric][i] for job_id in range(args.n_jobs)])
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
                # t_values = [id * n_per_job: (id + 1) * n_per_job]
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


def add_searchlight_permutation_args(parser):
    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

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
    parser.add_argument("--n-permutations-group-level", type=int, default=10000)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)
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
