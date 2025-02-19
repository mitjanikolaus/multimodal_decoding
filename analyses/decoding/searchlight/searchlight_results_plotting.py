import argparse
import warnings

import h5py
import numpy as np
from PIL import Image
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
import pickle

from tqdm import tqdm

from analyses.cluster_analysis import calc_significance_cutoff
from analyses.decoding.searchlight.searchlight import searchlight_mode_from_args
from analyses.decoding.searchlight.searchlight_permutation_testing import METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC, \
    METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC, load_per_subject_scores, CHANCE_VALUES, \
    load_null_distr_per_subject_scores, permutation_results_dir, \
    get_hparam_suffix, add_searchlight_permutation_args
from eval import ACC_CAPTIONS, ACC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS, ACC_IMAGES_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES
from utils import RESULTS_DIR, HEMIS, save_plot_and_crop_img, append_images

DEFAULT_VIEWS = ["lateral", "medial", "ventral", "posterior"]
COLORBAR_MAX = 0.8
COLORBAR_THRESHOLD_MIN = 0.55

COLORBAR_DIFFERENCE_MAX = 0.1

COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.02

CMAP = "cold_hot"
CMAP_POS_ONLY = "autumn"

DEFAULT_T_VALUE_THRESH = 1  # 0.824
DEFAULT_TFCE_VAL_THRESH = 10

PLOT_NULL_DISTR_NUM_SAMPLES = 10


def plot_test_statistics(test_statistics, args, results_path, subfolder=""):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    # if "t-values" in test_statistics:
    #     t_values = test_statistics['t-values']
    #     t_values_imgs_dir = str(os.path.join(results_path, "tmp", "t-values"))
    #     if subfolder:
    #         t_values_imgs_dir = os.path.join(t_values_imgs_dir, subfolder)
    #     os.makedirs(t_values_imgs_dir, exist_ok=True)
    #
    #     threshold = DEFAULT_T_VALUE_THRESH
    #     metrics = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES]
    #     print(f"plotting t values for {len(metrics)} metrics {subfolder}")
    #     cbar_max = {metric: None for metric in metrics}
    #     for metric in metrics:
    #         print(metric)
    #         for i, view in enumerate(args.views):
    #             for j, hemi in enumerate(HEMIS):
    #                 scores_hemi = t_values[hemi][metric]
    #                 if cbar_max[metric] is None:
    #                     cbar_max[metric] = np.nanmax(scores_hemi)
    #                 plotting.plot_surf_stat_map(
    #                     fsaverage[f"infl_{hemi}"],
    #                     scores_hemi,
    #                     hemi=hemi,
    #                     view=view,
    #                     bg_map=fsaverage[f"sulc_{hemi}"],
    #                     bg_on_data=True,
    #                     colorbar=False,
    #                     threshold=threshold,
    #                     vmax=cbar_max[metric],
    #                     vmin=0,
    #                     cmap=CMAP_POS_ONLY,
    #                 )
    #                 title = f"{metric}_{view}_{hemi}"
    #                 save_plot_and_crop_img(os.path.join(t_values_imgs_dir, f"{title}.png"))
    #
    #     # plot for cbar:
    #     plotting.plot_surf_stat_map(
    #         fsaverage[f"infl_{HEMIS[0]}"],
    #         t_values[HEMIS[0]][metrics[0]],
    #         hemi=HEMIS[0],
    #         view=args.views[0],
    #         bg_map=fsaverage[f"sulc_{HEMIS[0]}"],
    #         bg_on_data=True,
    #         colorbar=True,
    #         threshold=threshold,
    #         vmax=cbar_max[metrics[0]],
    #         vmin=0,
    #         cmap=CMAP_POS_ONLY,
    #     )
    #     save_plot_and_crop_img(os.path.join(t_values_imgs_dir, "colorbar.png"), crop_cbar=True)

    # plot remaining test stats
    test_statistics_filtered = test_statistics.copy()
    del test_statistics_filtered['t-values']

    # null_distribution_tfce_values_file = os.path.join(
    #     permutation_results_dir(args),
    #     f"tfce_values_null_distribution{get_hparam_suffix(args)}.p"
    # )
    # null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))
    # significance_cutoff, _ = calc_significance_cutoff(null_distribution_tfce_values, args.metric,
    #                                                   args.p_value_threshold)
    significance_cutoff = 251

    print(f"plotting test stats {subfolder}")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    cbar_max = {stat: None for stat in test_statistics.keys()}
    for (stat_name, values) in test_statistics.items():
        test_stat_imgs_dir = str(os.path.join(results_path, "tmp", f"{stat_name}"))
        if subfolder:
            test_stat_imgs_dir = os.path.join(test_stat_imgs_dir, subfolder)
        os.makedirs(test_stat_imgs_dir, exist_ok=True)

        threshold = DEFAULT_T_VALUE_THRESH if stat_name.startswith("t-values") else significance_cutoff
        for i, view in enumerate(args.views):
            for j, hemi in enumerate(HEMIS):
                scores_hemi = values[hemi][args.metric]
                if cbar_max[stat_name] is None:
                    if (stat_name == "t-values-smoothed") and (cbar_max['t-values'] is not None):
                        cbar_max[stat_name] = cbar_max['t-values']
                    else:
                        cbar_max[stat_name] = np.nanmax(scores_hemi)
                plotting.plot_surf_stat_map(
                    fsaverage[f"infl_{hemi}"],
                    scores_hemi,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    colorbar=False,
                    threshold=threshold,
                    vmax=cbar_max[stat_name],
                    vmin=0,
                    cmap=CMAP_POS_ONLY,
                )
                title = f"{args.metric}_{view}_{hemi}"
                save_plot_and_crop_img(os.path.join(test_stat_imgs_dir, f"{title}.png"))
        # plot for cbar:
        fig = plt.figure(figsize=(7, 6))
        plotting.plot_surf_stat_map(
            fsaverage[f"infl_{HEMIS[0]}"],
            values[HEMIS[0]][args.metric],
            hemi=HEMIS[0],
            view=args.views[0],
            bg_map=fsaverage[f"sulc_{HEMIS[0]}"],
            bg_on_data=True,
            colorbar=True,
            threshold=threshold,
            vmax=cbar_max[stat_name],
            vmin=0,
            cmap=CMAP_POS_ONLY,
            figure=fig,
        )
        save_plot_and_crop_img(os.path.join(test_stat_imgs_dir, f"colorbar_{args.metric}.png"), crop_cbar=True)


def plot_acc_scores(per_subject_scores, args, results_path, subfolder=""):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    metrics = [ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS, ACC_IMAGES_MOD_SPECIFIC_CAPTIONS,
               ACC_CAPTIONS_MOD_SPECIFIC_IMAGES]

    acc_scores_imgs_dir = str(os.path.join(results_path, "tmp", "acc_scores"))
    if subfolder:
        acc_scores_imgs_dir = os.path.join(acc_scores_imgs_dir, subfolder)
    os.makedirs(acc_scores_imgs_dir, exist_ok=True)

    print(f"plotting acc scores. {subfolder}")
    for metric in metrics:
        cbar_max = None
        threshold = COLORBAR_THRESHOLD_MIN
        if CHANCE_VALUES[metric] == 0:
            threshold = COLORBAR_DIFFERENCE_THRESHOLD_MIN

        for j, hemi in enumerate(HEMIS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                score_hemi_avgd = np.nanmean([per_subject_scores[subj][hemi][metric] for subj in args.subjects], axis=0)

            print(
                f"metric: {metric} {hemi} hemi mean: {np.nanmean(score_hemi_avgd):.2f} | "
                f"max: {np.nanmax(score_hemi_avgd):.2f}")

            for i, view in enumerate(args.views):
                if cbar_max is None:
                    cbar_max = min(np.nanmax(score_hemi_avgd), 99)

                plotting.plot_surf_stat_map(
                    fsaverage[f"infl_{hemi}"],
                    score_hemi_avgd,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    colorbar=False,
                    threshold=threshold,
                    vmax=COLORBAR_MAX if CHANCE_VALUES[metric] == 0.5 else COLORBAR_DIFFERENCE_MAX,
                    vmin=0.5 if CHANCE_VALUES[metric] == 0.5 else None,
                    cmap=CMAP_POS_ONLY if CHANCE_VALUES[metric] == 0.5 else CMAP,
                    symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
                )
                title = f"{metric}_{view}_{hemi}"
                save_plot_and_crop_img(os.path.join(acc_scores_imgs_dir, f"{title}.png"))

        plotting.plot_surf_stat_map(
            fsaverage[f"infl_{HEMIS[0]}"],
            score_hemi_avgd,
            hemi=HEMIS[0],
            view=args.views[0],
            bg_map=fsaverage[f"sulc_{HEMIS[0]}"],
            bg_on_data=True,
            colorbar=True,
            threshold=threshold,
            vmax=COLORBAR_MAX if CHANCE_VALUES[metric] == 0.5 else COLORBAR_DIFFERENCE_MAX,
            vmin=0.5 if CHANCE_VALUES[metric] == 0.5 else None,
            cmap=CMAP_POS_ONLY if CHANCE_VALUES[metric] == 0.5 else CMAP,
            symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
        )
        save_plot_and_crop_img(os.path.join(acc_scores_imgs_dir, f"colorbar_{metric}.png"), crop_cbar=True)


def plot_p_values(results_path, args):
    print(f"plotting (p-values)")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    p_values_path = os.path.join(permutation_results_dir(args), f"p_values{get_hparam_suffix(args)}.p")
    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    cbar_max = np.nanmax(np.concatenate((p_values['left'], p_values['right'])))
    cbar_min = 0
    p_values_imgs_dir = str(os.path.join(results_path, "tmp", "p_values"))
    os.makedirs(p_values_imgs_dir, exist_ok=True)

    for i, view in enumerate(args.views):
        for j, hemi in enumerate(HEMIS):
            scores_hemi = p_values[hemi]
            plotting.plot_surf_stat_map(
                fsaverage[f"infl_{hemi}"],
                scores_hemi,
                hemi=hemi,
                view=view,
                bg_map=fsaverage[f"sulc_{hemi}"],
                bg_on_data=True,
                colorbar=False,
                threshold=-np.log10(args.p_value_threshold),
                vmax=cbar_max,
                vmin=cbar_min,
                cmap=CMAP_POS_ONLY,
                symmetric_cbar=False,
            )
            title = f"{view}_{hemi}"
            save_plot_and_crop_img(os.path.join(p_values_imgs_dir, f"{title}.png"))
    # plot for cbar:
    plotting.plot_surf_stat_map(
        fsaverage[f"infl_{HEMIS[0]}"],
        p_values[HEMIS[0]],
        hemi=HEMIS[0],
        view=args.views[0],
        bg_map=fsaverage[f"sulc_{HEMIS[0]}"],
        bg_on_data=True,
        colorbar=True,
        threshold=-np.log10(args.p_value_threshold),
        vmax=cbar_max,
        vmin=cbar_min,
        cmap=CMAP_POS_ONLY,
        symmetric_cbar=False,
    )
    save_plot_and_crop_img(os.path.join(p_values_imgs_dir, "colorbar.png"), crop_cbar=True)


def create_composite_image(args):
    results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution, searchlight_mode_from_args(args)))

    tfce_values_img_dir = str(os.path.join(results_path, "tmp", "tfce-values"))
    tfce_val_img = Image.open(os.path.join(tfce_values_img_dir, f"{args.metric}_medial_left.png"))
    # offset_size = (int(p_val_img.size[0]/10), p_val_img.size[1])
    # image_whitespace = Image.new('RGBA', offset_size, color=(255, 255, 255, 0))
    cbar = Image.open(os.path.join(tfce_values_img_dir, f"colorbar_{args.metric}.png"))
    tfce_val_img = append_images([cbar, tfce_val_img], padding=150)  # image_whitespace
    tfce_val_img = tfce_val_img.resize((int(tfce_val_img.size[0] * 1.1), int(tfce_val_img.size[1] * 1.1)))

    acc_scores_imgs_dir = str(os.path.join(results_path, "tmp", "acc_scores"))
    acc_scores_imgs = []
    metrics = [ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS, ACC_IMAGES_MOD_SPECIFIC_CAPTIONS,
               ACC_CAPTIONS_MOD_SPECIFIC_IMAGES]
    for metric in metrics:
        images = Image.open(os.path.join(acc_scores_imgs_dir, f"{metric}_medial_left.png"))
        cbar = Image.open(os.path.join(acc_scores_imgs_dir, f"colorbar_{metric}.png"))
        if metric in [ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS]:
            acc_scores_img = append_images([cbar, images], padding=50)
        else:
            acc_scores_img = append_images([images, cbar], padding=50)

        acc_scores_img = acc_scores_img.resize((int(acc_scores_img.size[0] / 1.2), int(acc_scores_img.size[1] / 1.2)))
        acc_scores_imgs.append(acc_scores_img)

    acc_scores_imgs_acc = append_images(acc_scores_imgs[:2], horizontally=False, padding=300)
    acc_scores_imgs_diff = append_images(acc_scores_imgs[2:], horizontally=False, padding=300)

    acc_imgs = append_images([acc_scores_imgs_acc, acc_scores_imgs_diff], padding=400)

    full_img = append_images([acc_imgs, tfce_val_img], horizontally=False, padding=400)

    path = os.path.join(results_path, "searchlight_methods.png")
    full_img.save(path, transparent=True)
    print("done")


def run(args):
    results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution, searchlight_mode_from_args(args)))
    os.makedirs(results_path, exist_ok=True)

    plot_p_values(results_path, args)

    per_subject_scores = load_per_subject_scores(args)
    plot_acc_scores(per_subject_scores, args, results_path)

    t_values_path = os.path.join(permutation_results_dir(args), "t_values.p")
    test_statistics = {"t-values": pickle.load(open(t_values_path, 'rb'))}
    tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
    test_statistics["tfce-values"] = pickle.load(open(tfce_values_path, 'rb'))
    plot_test_statistics(test_statistics, args, results_path)

    create_composite_image(args)

    if args.plot_null_distr:
        print("plotting acc maps for null distribution examples")
        per_subject_scores_null_distr = load_null_distr_per_subject_scores(args)
        for i in range(PLOT_NULL_DISTR_NUM_SAMPLES):
            plot_acc_scores(per_subject_scores_null_distr[i], args, results_path, subfolder=f"_null_distr_{i}")

        print("plotting test stats for null distribution examples")
        t_values_null_distribution_path = os.path.join(
            permutation_results_dir(args), f"t_values_null_distribution.hdf5"
        )
        with h5py.File(t_values_null_distribution_path, 'r') as null_distribution_t_values:
            t_values_smooth_null_distribution = None
            null_distribution_tfce_values_file = os.path.join(
                permutation_results_dir(args),
                f"tfce_values_null_distribution{get_hparam_suffix(args)}.p"
            )
            null_distribution_test_statistic = pickle.load(open(null_distribution_tfce_values_file, 'rb'))

            for i in range(PLOT_NULL_DISTR_NUM_SAMPLES):
                test_statistics = {"t-values": null_distribution_t_values[i]}
                if t_values_smooth_null_distribution is not None:
                    test_statistics["t-values-smoothed"] = t_values_smooth_null_distribution[i]
                test_statistics["tfce-values"] = null_distribution_test_statistic[i]
                plot_test_statistics(test_statistics, args, results_path, subfolder=f"_null_distr_{i}")

    if args.per_subject_plots:
        metrics = [ACC_IMAGES_MOD_SPECIFIC_IMAGES, ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS, ACC_IMAGES_MOD_SPECIFIC_CAPTIONS,
                   ACC_CAPTIONS_MOD_SPECIFIC_IMAGES]
        print("\n\nCreating per-subject plots..")
        for subject, scores in tqdm(per_subject_scores.items()):
            fig = plt.figure(figsize=(5 * len(args.views), len(metrics) * 2))
            subfigs = fig.subfigures(nrows=len(metrics), ncols=1)
            fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

            for subfig, metric in zip(subfigs, metrics):
                subfig.suptitle(f'{metric}', x=0, horizontalalignment="left")
                axes = subfig.subplots(nrows=1, ncols=2 * len(args.views), subplot_kw={'projection': '3d'})
                cbar_max = None
                cbar_min = None
                for i, view in enumerate(args.views):
                    for j, hemi in enumerate(['left', 'right']):
                        scores_hemi = scores[hemi][metric]

                        infl_mesh = fsaverage[f"infl_{hemi}"]
                        if cbar_max is None:
                            cbar_max = np.nanmax(scores_hemi)
                            cbar_min = np.nanmin(scores_hemi)

                        plotting.plot_surf_stat_map(
                            infl_mesh,
                            scores_hemi,
                            hemi=hemi,
                            view=view,
                            bg_map=fsaverage[f"sulc_{hemi}"],
                            bg_on_data=True,
                            axes=axes[i * 2 + j],
                            colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                            threshold=COLORBAR_THRESHOLD_MIN if cbar_min >= 0 else COLORBAR_DIFFERENCE_THRESHOLD_MIN,
                            vmax=COLORBAR_MAX if cbar_min >= 0 else None,  # cbar_max,
                            vmin=0.5 if cbar_min >= 0 else None,
                            cmap=CMAP_POS_ONLY if cbar_min >= 0 else CMAP,
                            symmetric_cbar=True if cbar_min < 0 else "auto",
                        )
                        axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)

            title = f"{subject}"
            fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
            results_searchlight = os.path.join(results_path, f"{title}.png")
            plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
            plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    parser.add_argument("--per-subject-plots", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot-null-distr", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)

    parser.add_argument("--plot-n-neighbors-correlation-graph", action="store_true", default=False)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
