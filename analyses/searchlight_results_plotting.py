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
from analyses.searchlight_permutation_testing import METRIC_MIN_DIFF_BOTH_MODALITIES, METRIC_DIFF_IMAGES, \
    METRIC_DIFF_CAPTIONS, METRIC_CAPTIONS, METRIC_IMAGES
from utils import RESULTS_DIR, SUBJECTS, HEMIS


VIEWS = ["lateral", "medial", "ventral"]


def run(args):
    p_values_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
                                 f"p_values_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p")
    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    p_values['left'][p_values['left'] > 0] = -np.log10(
        p_values['left'][p_values['left'] > 0])
    p_values['right'][p_values['right'] > 0] = -np.log10(
        p_values['right'][p_values['right'] > 0])

    print(f"plotting (p-values)")
    metric = METRIC_MIN_DIFF_BOTH_MODALITIES
    fig = plt.figure(figsize=(5 * len(VIEWS), 2))
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    fig.suptitle(f'{metric}', x=0, horizontalalignment="left")
    axes = fig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
    cbar_max = None
    for i, view in enumerate(VIEWS):
        for j, hemi in enumerate(HEMIS):
            scores_hemi = p_values[hemi]
            infl_mesh = fsaverage[f"infl_{hemi}"]
            if cbar_max is None:
                cbar_max = min(np.nanmax(scores_hemi), 99)
                print("cbar max: ", cbar_max)
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
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()

    t_values_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
                                 "t_values.p")
    t_values = pickle.load(open(t_values_path, 'rb'))

    print(f"plotting (t-values) threshold {0.5}")
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
                        threshold=0.5 if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else 2.015,
                        vmax=cbar_max,
                        vmin=0 if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else -cbar_max,
                        cmap="hot" if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else "cold_hot",
                        symmetric_cbar=False if metric == METRIC_MIN_DIFF_BOTH_MODALITIES else True,
                    )
                    axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)
                else:
                    axes[i * 2 + j].axis('off')
    title = f"{args.model}_{args.mode}_group_level_t_values_tresh_{0.5}"
    # fig.suptitle(title)
    # fig.tight_layout()
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
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


def calc_t_values_null_distr():
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

    for path_agnostic, path_caps, path_imgs in zip(paths_mod_agnostic, paths_mod_specific_captions,
                                                   paths_mod_specific_images):
        print('loading: ', path_agnostic)
        hemi = os.path.dirname(path_agnostic).split("/")[-2]
        subject = os.path.dirname(path_agnostic).split("/")[-4]

        results_agnostic = pickle.load(open(path_agnostic, 'rb'))
        nan_locations = results_agnostic['nan_locations']

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
            if len(per_subject_scores_null_distr) <= i:
                per_subject_scores_null_distr.append({subj: dict() for subj in SUBJECTS})
            scores = process_scores(distr, distr_caps, distr_imgs, nan_locations)
            per_subject_scores_null_distr[i][subject][hemi] = scores

    def shuffle_and_calc_t_values(per_subject_scores, proc_id, n_iters_per_job):
        job_t_vals = []
        iterator = tqdm(range(n_iters_per_job)) if proc_id == 0 else range(n_iters_per_job)
        for _ in iterator:
            t_values = {hemi: dict() for hemi in HEMIS}
            for hemi in HEMIS:
                t_vals = dict()

                for metric in [METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS]:
                    random_idx = np.random.choice(len(per_subject_scores), size=len(SUBJECTS))
                    data = np.array(
                        [per_subject_scores[idx][subj][hemi][metric] for idx, subj in
                         zip(random_idx, SUBJECTS)])
                    popmean = CHANCE_VALUES[metric]
                    t_vals[metric] = calc_image_t_values(data, popmean)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = np.nanmin(
                        (t_vals[METRIC_DIFF_CAPTIONS], t_vals[METRIC_DIFF_IMAGES]),
                        axis=0)

            job_t_vals.append(t_values)
        return job_t_vals

    n_iters_per_job = math.ceil(args.n_permutations_group_level / args.n_jobs)
    n_iters_last_job = n_iters_per_job
    if args.n_permutations_group_level % args.n_jobs != 0:
        n_iters_last_job = args.n_permutations_group_level - (n_iters_per_job * args.n_jobs)
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
        t_values_null_distribution = calc_t_values_null_distr()
        os.makedirs(os.path.dirname(t_values_null_distribution_path), exist_ok=True)
        pickle.dump(t_values_null_distribution, open(t_values_null_distribution_path, 'wb'))
    else:
        t_values_null_distribution = pickle.load(open(t_values_null_distribution_path, 'rb'))

    if args.smoothing_iterations > 0:
        smooth_t_values_null_distribution_path = os.path.join(
            SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
            args.resolution,
            args.mode, f"t_values_null_distribution_smoothed_{args.smoothing_iterations}.p"
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
                            t_vals[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES],
                            match=None,
                            iterations=args.smoothing_iterations
                        )
                        t_vals[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES] = smoothed
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
            t_values_null_distribution = np.concatenate(all_smooth_t_vals)
            pickle.dump(t_values_null_distribution, open(smooth_t_values_null_distribution_path, 'wb'))
        else:
            t_values_null_distribution = pickle.load(open(smooth_t_values_null_distribution_path, 'rb'))

    if args.tfce:
        tfce_values_null_distribution_path = os.path.join(
            SEARCHLIGHT_OUT_DIR, "train", args.model, args.features,
            args.resolution,
            args.mode,
            f"tfce_values_null_distribution_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
        )
        if not os.path.isfile(tfce_values_null_distribution_path):
            print(f"Calculating tfce values for null distribution")

            edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)

            def tfce_values_job(t_values, edge_lengths, proc_id):
                iterator = tqdm(t_values) if proc_id == 0 else t_values
                tfce_values = [
                    calc_tfce_values(vals, edge_lengths, h=args.tfce_h, e=args.tfce_e) for vals in
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
    parser.add_argument("--per-subject-plots", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--smoothing-iterations", type=int, default=0)

    parser.add_argument("--tfce", action="store_true")
    parser.add_argument("--tfce-h", type=float, default=2)
    parser.add_argument("--tfce-e", type=float, default=1)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-permutations-group-level", type=int, default=10000)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.features = get_default_features(args.model) if args.features == FEATS_SELECT_DEFAULT else args.features

    create_null_distribution(args)
    run(args)
