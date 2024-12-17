import argparse
import hashlib
import itertools
import math
import warnings

import h5py
import numpy as np
from joblib import Parallel, delayed
import os
import pickle

from tqdm import tqdm, trange

from analyses.ridge_regression_decoding import get_default_vision_features, get_default_lang_features
from analyses.ridge_regression_encoding import ENCODING_RESULTS_DIR, get_null_distr_results_path, get_results_file_path
from analyses.searchlight.searchlight_permutation_testing import get_edge_lengths_dicts_based_on_edges, \
    calc_tfce_values, calc_significance_cutoff, create_masks
from utils import SUBJECTS, HEMIS, DEFAULT_RESOLUTION, \
    METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, MODE_AGNOSTIC, MOD_SPECIFIC_IMAGES, MOD_SPECIFIC_CAPTIONS, \
    CORR_CAPTIONS, CORR_IMAGES, METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, CORR_CROSS_IMAGES_TO_CAPTIONS, \
    CORR_CROSS_CAPTIONS_TO_IMAGES, METRIC_CROSS_ENCODING, FS_HEMI_NAMES, export_to_gifti, \
    METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC_ALT, METRIC_CROSS_ENCODING_ALT

DEFAULT_N_JOBS = 3
ENCODING_PERMUTATION_TESTING_RESULTS_DIR = os.path.join(ENCODING_RESULTS_DIR, "permutation_testing")

CORR_IMAGES_MOD_SPECIFIC_IMAGES = 'corr_images_mod_specific_images'
CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS = 'corr_captions_mod_specific_captions'


def load_per_subject_scores(args, return_nan_locations=False):
    print("loading per-subject scores")

    per_subject_scores = {subj: dict() for subj in args.subjects}
    per_subject_nan_locations = {subj: dict() for subj in args.subjects}

    for subject in tqdm(args.subjects):
        for hemi in HEMIS:
            vision_features = get_default_vision_features(args.model)
            lang_features = get_default_lang_features(args.model)
            results_agnostic_file = get_results_file_path(subject, MODE_AGNOSTIC, args.model, args.features,
                                                          vision_features, lang_features, args.resolution, hemi)

            vision_features = get_default_vision_features(args.mod_specific_vision_model)
            lang_features = get_default_lang_features(args.mod_specific_vision_model)
            results_mod_specific_vision_file = get_results_file_path(subject, MOD_SPECIFIC_IMAGES,
                                                                     args.mod_specific_vision_model,
                                                                     args.mod_specific_vision_features,
                                                                     vision_features, lang_features,
                                                                     args.resolution,
                                                                     hemi)
            vision_features = get_default_vision_features(args.mod_specific_lang_model)
            lang_features = get_default_lang_features(args.mod_specific_lang_model)
            results_mod_specific_lang_file = get_results_file_path(subject, MOD_SPECIFIC_CAPTIONS,
                                                                   args.mod_specific_lang_model,
                                                                   args.mod_specific_lang_features,
                                                                   vision_features, lang_features,
                                                                   args.resolution,
                                                                   hemi)

            scores_agnostic = pickle.load(open(results_agnostic_file, 'rb'))
            nan_locations = scores_agnostic['nan_locations']
            per_subject_nan_locations[subject][hemi] = nan_locations

            if os.path.isfile(results_mod_specific_vision_file):
                scores_images = pickle.load(open(results_mod_specific_vision_file, 'rb'))
            else:
                print(f"Missing modality-specific results: {results_mod_specific_vision_file}")
                scores_images = None

            if os.path.isfile(results_mod_specific_lang_file):
                scores_captions = pickle.load(open(results_mod_specific_lang_file, 'rb'))
            else:
                print(f"Missing modality-specific results: {results_mod_specific_lang_file}")
                scores_captions = None

            scores = process_scores(scores_agnostic, scores_captions, scores_images, nan_locations)

            # print({n: round(np.nanmean(score), 4) for n, score in scores.items()})
            # print({f"{n}_max": round(np.nanmax(score), 2) for n, score in scores.items()})
            # print("")

            per_subject_scores[subject][hemi] = scores

    if return_nan_locations:
        return per_subject_scores, per_subject_nan_locations
    else:
        return per_subject_scores


def calc_image_t_values(data, t_vals_cache=None, precision=2, epsilon=1e-8):
    # data = data.round(precision)
    # iterator = tqdm(data.T) if use_tqdm else data.T
    t_vals = data.mean(axis=0)
    # if t_vals_cache is None:
    #     return np.array(
    #         [calc_t_value(x, popmean, epsilon) for x in iterator]
    #     )
    # else:
    #     t_vals = []
    #     for x in iterator:
    #         x_no_nan = x[~np.isnan(x)]
    #         if x_no_nan.mean() > popmean:
    #             key = hashlib.sha1(np.sort(x_no_nan)).hexdigest()
    #             if key in t_vals_cache:
    #                 t_vals.append(t_vals_cache[key])
    #             else:
    #                 if np.all(x_no_nan == x_no_nan[0]):
    #                     # Add/subtract epsilon for numerical stability
    #                     x_no_nan[0] = x_no_nan[0] + epsilon
    #                     x_no_nan[-1] = x_no_nan[-1] - epsilon
    #                 t_val = stats.ttest_1samp(x_no_nan, popmean=popmean, alternative="greater")[0]
    #                 if np.isinf(t_val):
    #                     print(f"Inf t-val for values: {x_no_nan}")
    #                 t_vals.append(t_val)
    #                 t_vals_cache[key] = t_val
    #         else:
    #             # mean is below popmean, t value won't be significant
    #             t_vals.append(0)
    #
    #     return np.array(t_vals)


def calc_t_values(per_subject_scores):
    t_values = {hemi: dict() for hemi in HEMIS}
    for hemi in HEMIS:
        for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, CORR_IMAGES, CORR_CAPTIONS,
                       CORR_CROSS_IMAGES_TO_CAPTIONS, CORR_CROSS_CAPTIONS_TO_IMAGES, CORR_IMAGES_MOD_SPECIFIC_IMAGES,
                       CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS]:
            data = np.array([per_subject_scores[subj][hemi][metric] for subj in args.subjects])
            enough_data = np.argwhere(((~np.isnan(data)).sum(axis=0)) > 2)[:, 0]  # at least 3 datapoints
            t_values[hemi][metric] = np.repeat(np.nan, data.shape[1])
            t_values[hemi][metric][enough_data] = np.nanmean(data[:, enough_data], axis=0)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=RuntimeWarning)
        #     t_values[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC] = np.nanmin(
        #         (
        #             t_values[hemi][METRIC_DIFF_CAPTIONS],
        #             t_values[hemi][METRIC_DIFF_IMAGES],
        #             t_values[hemi][CORR_IMAGES],
        #             t_values[hemi][CORR_CAPTIONS]),
        #         axis=0)
        #     t_values[hemi][METRIC_CROSS_ENCODING] = np.nanmin(
        #         (t_values[hemi][CORR_CROSS_CAPTIONS_TO_IMAGES],
        #          t_values[hemi][CORR_CROSS_IMAGES_TO_CAPTIONS]),
        #         axis=0
        #     )

    return t_values


def calculate_metric_from_t_vals(t_vals, metric):
    for hemi in HEMIS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if metric == METRIC_CROSS_ENCODING:
                t_vals[hemi][metric] = np.nanmin(
                    (
                        t_vals[hemi][CORR_CROSS_CAPTIONS_TO_IMAGES],
                        t_vals[hemi][CORR_CROSS_IMAGES_TO_CAPTIONS]),
                    axis=0)
            elif metric == METRIC_CROSS_ENCODING_ALT:
                t_vals[hemi][metric] = np.nanmax(
                    (
                        t_vals[hemi][CORR_CROSS_CAPTIONS_TO_IMAGES],
                        t_vals[hemi][CORR_CROSS_IMAGES_TO_CAPTIONS]),
                    axis=0)
            elif metric == METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC:
                t_vals[hemi][metric] = np.nanmin(
                    (
                        t_vals[hemi][METRIC_DIFF_CAPTIONS],
                        t_vals[hemi][METRIC_DIFF_IMAGES],
                        t_vals[hemi][CORR_IMAGES],
                        t_vals[hemi][CORR_CAPTIONS]),
                    axis=0)
            elif metric == METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC_ALT:
                t_vals[hemi][metric] = np.nanmin(
                    (
                        np.nanmax((t_vals[hemi][METRIC_DIFF_CAPTIONS],
                                   t_vals[hemi][METRIC_DIFF_IMAGES]), axis=0),
                        t_vals[hemi][CORR_IMAGES],
                        t_vals[hemi][CORR_CAPTIONS]),
                    axis=0)
            else:
                raise RuntimeError(f"Unknown metric: {metric}")
    return t_vals


def calc_test_statistics(null_distr_tfce_values, args):
    t_values_path = os.path.join(permutation_results_dir(args), "t_values.p")
    per_subject_scores = load_per_subject_scores(args)
    print(f"Calculating t-values")
    t_values = calc_t_values(per_subject_scores)
    t_values = calculate_metric_from_t_vals(t_values, args.metric)
    pickle.dump(t_values, open(t_values_path, 'wb'))

    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
    if not os.path.isfile(tfce_values_path):
        print("calculating tfce..")
        edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)
        tfce_values = calc_tfce_values(t_values, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e,
                                       dh=args.tfce_dh, clip_value=args.tfce_clip)
        pickle.dump(tfce_values, open(tfce_values_path, "wb"))
    else:
        tfce_values = pickle.load(open(tfce_values_path, 'rb'))

    for hemi in HEMIS:
        print(f"mean tfce value ({hemi} hemi): {np.nanmean(tfce_values[hemi][args.metric]):.3f} | ", end="")
        print(f"max tfce value ({hemi} hemi): {np.nanmax(tfce_values[hemi][args.metric]):.3f}")

    significance_cutoff, max_test_statistic_distr = calc_significance_cutoff(null_distr_tfce_values, args.metric,
                                                                             args.p_value_threshold)

    p_values = {hemi: np.repeat(np.nan, t_values[hemi][CORR_IMAGES].shape) for hemi, t_vals in t_values.items()}
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


def process_scores(scores_agnostic, scores_mod_specific_captions, scores_mod_specific_images, nan_locations):
    scores = dict()
    for metric in [CORR_CAPTIONS, CORR_IMAGES]:
        scores[metric] = np.repeat(np.nan, nan_locations.shape)
        scores[metric][~nan_locations] = scores_agnostic[metric]

    if scores_mod_specific_captions is not None and scores_mod_specific_images is not None:
        scores_specific_captions = dict()
        for metric in [CORR_CAPTIONS, CORR_IMAGES]:
            scores_specific_captions[metric] = np.repeat(np.nan, nan_locations.shape)
            scores_specific_captions[metric][~nan_locations] = scores_mod_specific_captions[metric]

        scores_specific_images = dict()
        for metric in [CORR_CAPTIONS, CORR_IMAGES]:
            scores_specific_images[metric] = np.repeat(np.nan, nan_locations.shape)
            scores_specific_images[metric][~nan_locations] = scores_mod_specific_images[metric]

        scores[CORR_CROSS_CAPTIONS_TO_IMAGES] = np.repeat(np.nan, nan_locations.shape)
        scores[CORR_CROSS_CAPTIONS_TO_IMAGES][~nan_locations] = scores_mod_specific_captions[
            CORR_CROSS_CAPTIONS_TO_IMAGES]

        scores[CORR_CROSS_IMAGES_TO_CAPTIONS] = np.repeat(np.nan, nan_locations.shape)
        scores[CORR_CROSS_IMAGES_TO_CAPTIONS][~nan_locations] = scores_mod_specific_images[
            CORR_CROSS_IMAGES_TO_CAPTIONS]

        scores[METRIC_DIFF_IMAGES] = np.array(
            [ai - si for ai, si in zip(scores[CORR_IMAGES], scores_specific_images[CORR_IMAGES])]
        )
        scores[METRIC_DIFF_CAPTIONS] = np.array(
            [ac - sc for ac, sc in zip(scores[CORR_CAPTIONS], scores_specific_captions[CORR_CAPTIONS])]
        )

        scores[CORR_IMAGES_MOD_SPECIFIC_IMAGES] = scores_specific_images[CORR_IMAGES]
        scores[CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS] = scores_specific_captions[CORR_CAPTIONS]

    return scores


def load_null_distr_per_subject_scores(args):
    per_subject_scores_null_distr = []

    for subject in args.subjects:
        print(subject)
        for hemi in HEMIS:
            vision_features = get_default_vision_features(args.model)
            lang_features = get_default_lang_features(args.model)
            null_distr_agnostic_file = get_null_distr_results_path(subject, MODE_AGNOSTIC, args.model, args.features,
                                                                   vision_features, lang_features, args.resolution,
                                                                   hemi)

            vision_features = get_default_vision_features(args.mod_specific_vision_model)
            lang_features = get_default_lang_features(args.mod_specific_vision_model)
            null_distr_results_mod_specific_vision_file = get_null_distr_results_path(subject, MOD_SPECIFIC_IMAGES,
                                                                                      args.mod_specific_vision_model,
                                                                                      args.mod_specific_vision_features,
                                                                                      vision_features, lang_features,
                                                                                      args.resolution,
                                                                                      hemi)
            vision_features = get_default_vision_features(args.mod_specific_lang_model)
            lang_features = get_default_lang_features(args.mod_specific_lang_model)
            null_distr_results_mod_specific_lang_file = get_null_distr_results_path(subject, MOD_SPECIFIC_CAPTIONS,
                                                                                    args.mod_specific_lang_model,
                                                                                    args.mod_specific_lang_features,
                                                                                    vision_features, lang_features,
                                                                                    args.resolution,
                                                                                    hemi)

            null_distribution_agnostic = pickle.load(open(null_distr_agnostic_file, 'rb'))
            null_distribution_images = pickle.load(open(null_distr_results_mod_specific_vision_file, 'rb'))
            null_distribution_captions = pickle.load(open(null_distr_results_mod_specific_lang_file, 'rb'))

            results_agnostic_file = get_results_file_path(subject, MODE_AGNOSTIC, args.model, args.features,
                                                          vision_features, lang_features, args.resolution, hemi)
            nan_locations = pickle.load(open(results_agnostic_file, 'rb'))['nan_locations']

            num_permutations = len(null_distribution_agnostic)
            for i in tqdm(range(num_permutations), desc='final per subject scores null distribution dict creation'):
                scores = process_scores(null_distribution_agnostic[i], null_distribution_captions[i],
                                        null_distribution_images[i], nan_locations)
                if len(per_subject_scores_null_distr) <= i:
                    per_subject_scores_null_distr.append({subj: dict() for subj in args.subjects})
                per_subject_scores_null_distr[i][subject][hemi] = scores
    return per_subject_scores_null_distr


def calc_t_values_null_distr(args, out_path):
    per_subject_scores_null_distr_path = os.path.join(
        permutation_results_dir(args), f"per_subject_scores_null_distr.p"
    )
    if not os.path.isfile(per_subject_scores_null_distr_path):
        print("loading per subject null distr scores")
        per_subject_scores = load_null_distr_per_subject_scores(args)
        os.makedirs(os.path.dirname(per_subject_scores_null_distr_path), exist_ok=True)
        pickle.dump(per_subject_scores, open(per_subject_scores_null_distr_path, 'wb'))
    else:
        print("loading precomputed per subject null distr scores..", end=' ')
        per_subject_scores = pickle.load(open(per_subject_scores_null_distr_path, 'rb'))
        print('done.')

    permutations_iter = itertools.permutations(range(len(per_subject_scores)), len(args.subjects))
    permutations = [next(permutations_iter) for _ in range(args.n_permutations_group_level)]
    with h5py.File(out_path, 'w') as all_t_vals_file:
        dsets = dict()
        for hemi in HEMIS:
            print(hemi)
            n_vertices = per_subject_scores[0][args.subjects[0]][hemi][CORR_IMAGES].size
            tvals_shape = (args.n_permutations_group_level, n_vertices)
            dsets[hemi] = dict()

            for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, CORR_IMAGES, CORR_CAPTIONS,
                           CORR_CROSS_IMAGES_TO_CAPTIONS, CORR_CROSS_CAPTIONS_TO_IMAGES,
                           # METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_ENCODING
                           ]:
                dsets[hemi][metric] = all_t_vals_file.create_dataset(f"{hemi}__{metric}", tvals_shape, dtype='float32')

            for perm_idx in tqdm(range(args.n_permutations_group_level)):
                tvals = dict()
                for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS, CORR_IMAGES, CORR_CAPTIONS,
                               CORR_CROSS_IMAGES_TO_CAPTIONS, CORR_CROSS_CAPTIONS_TO_IMAGES,
                               ]:
                    data = [per_subject_scores[idx][subj][hemi][metric] for idx, subj in
                            zip(permutations[perm_idx], args.subjects)]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        tvals[metric] = np.nanmean(data, axis=0)
                    dsets[hemi][metric][perm_idx] = tvals[metric]

                # with warnings.catch_warnings():
                #     warnings.simplefilter("ignore", category=RuntimeWarning)
                #     dsets[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC][perm_idx] = np.nanmin(
                #         (
                #             tvals[METRIC_DIFF_CAPTIONS],
                #             tvals[METRIC_DIFF_IMAGES],
                #             tvals[CORR_IMAGES],
                #             tvals[CORR_CAPTIONS]),
                #         axis=0)
                #     dsets[hemi][METRIC_CROSS_ENCODING][perm_idx] = np.nanmin(
                #         (tvals[CORR_CROSS_IMAGES_TO_CAPTIONS],
                #          tvals[CORR_CROSS_CAPTIONS_TO_IMAGES]),
                #         axis=0
                #     )


def permutation_results_dir(args):
    return str(os.path.join(
        ENCODING_PERMUTATION_TESTING_RESULTS_DIR, args.model, args.features, args.mod_specific_vision_model,
        args.mod_specific_vision_features, args.mod_specific_lang_model, args.mod_specific_lang_features,
        args.resolution,
    ))


def get_hparam_suffix(args):
    return f"_{args.metric}_h_{args.tfce_h}_e_{args.tfce_e}_dh_{args.tfce_dh}_clip_{args.tfce_clip}"


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
            indices = range(proc_id * n_per_job, min((proc_id + 1) * n_per_job, args.n_permutations_group_level))
            iterator = tqdm(indices) if proc_id == 0 else indices

            with h5py.File(t_vals_null_distr_path, 'r') as t_vals:
                values = dict()
                for hemi in HEMIS:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        if args.metric == METRIC_CROSS_ENCODING:
                            values[hemi] = [np.nanmin(
                                (
                                    t_vals[f"{hemi}__{CORR_CROSS_CAPTIONS_TO_IMAGES}"][perm_idx],
                                    t_vals[f"{hemi}__{CORR_CROSS_IMAGES_TO_CAPTIONS}"][perm_idx]),
                                axis=0) for perm_idx in iterator]
                        elif args.metric == METRIC_CROSS_ENCODING_ALT:
                            values[hemi] = [np.nanmax(
                                (
                                    t_vals[f"{hemi}__{CORR_CROSS_CAPTIONS_TO_IMAGES}"][perm_idx],
                                    t_vals[f"{hemi}__{CORR_CROSS_IMAGES_TO_CAPTIONS}"][perm_idx]),
                                axis=0) for perm_idx in iterator]
                        elif args.metric == METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC:
                            values[hemi] = [np.nanmin(
                                (
                                    t_vals[f"{hemi}__{METRIC_DIFF_CAPTIONS}"][perm_idx],
                                    t_vals[f"{hemi}__{METRIC_DIFF_IMAGES}"][perm_idx],
                                    t_vals[f"{hemi}__{CORR_IMAGES}"][perm_idx],
                                    t_vals[f"{hemi}__{CORR_CAPTIONS}"][perm_idx]),
                                axis=0) for perm_idx in iterator]
                        elif args.metric == METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC_ALT:
                            values[hemi] = [np.nanmin(
                                (
                                    np.nanmax((t_vals[f"{hemi}__{METRIC_DIFF_CAPTIONS}"][perm_idx],
                                               t_vals[f"{hemi}__{METRIC_DIFF_IMAGES}"][perm_idx]), axis=0),
                                    t_vals[f"{hemi}__{CORR_IMAGES}"][perm_idx],
                                    t_vals[f"{hemi}__{CORR_CAPTIONS}"][perm_idx]),
                                axis=0) for perm_idx in iterator]
                        else:
                            raise RuntimeError(f"Unknown metric: {args.metric}")

                tfce_values = []
                iterator = trange(len(values[HEMIS[0]])) if proc_id == 0 else range(len(values[HEMIS[0]]))
                for iteration in iterator:
                    vals = {hemi: {args.metric: values[hemi][iteration]} for hemi in HEMIS}
                    tfce_values.append(
                        calc_tfce_values(
                            vals, edge_lengths, args.metric, h=args.tfce_h, e=args.tfce_e, dh=args.tfce_dh,
                            clip_value=args.tfce_clip
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
    else:
        print(tfce_values_null_distribution_path)
        tfce_values = pickle.load(open(tfce_values_null_distribution_path, 'rb'))
    return tfce_values


def create_t_value_maps(results_dir):
    print("Creating t value gifti results masks")
    t_values_path = os.path.join(permutation_results_dir(args), "t_values.p")

    results_maps_path = os.path.join(results_dir, "results_maps")
    os.makedirs(results_maps_path, exist_ok=True)

    t_values = pickle.load(open(t_values_path, "rb"))

    for hemi in HEMIS:
        for metric in [CORR_IMAGES, CORR_CAPTIONS, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES,
                       CORR_CROSS_IMAGES_TO_CAPTIONS, CORR_CROSS_CAPTIONS_TO_IMAGES,
                       CORR_IMAGES_MOD_SPECIFIC_IMAGES, CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS]:
            path_out = os.path.join(results_maps_path, f"t_values_{metric}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(t_values[hemi][metric], path_out)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--model", type=str, default='imagebind')
    parser.add_argument("--features", type=str, default="avg_test_avg")

    parser.add_argument("--mod-specific-vision-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-vision-features", type=str, default="vision_test_vision")

    parser.add_argument("--mod-specific-lang-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-lang-features", type=str, default="lang_test_lang")

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)
    parser.add_argument("--tfce-dh", type=float, default=0.01)
    parser.add_argument("--tfce-clip", type=float, default=100)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-permutations-group-level", type=int, default=10000)

    parser.add_argument("--metric", type=str, default=METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(ENCODING_PERMUTATION_TESTING_RESULTS_DIR, exist_ok=True)
    args = get_args()

    print(f"\n\nPermutation Testing for {args.metric}\n")
    null_distr_tfce_values = create_null_distribution(args)
    calc_test_statistics(null_distr_tfce_values, args)

    create_masks(permutation_results_dir(args), args.metric, args.p_value_threshold, get_hparam_suffix(args),
                 args.resolution)

    create_t_value_maps(permutation_results_dir(args))
