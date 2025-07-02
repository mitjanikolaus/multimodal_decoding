import argparse
import warnings

import numpy as np
from PIL import Image
from nilearn import datasets, plotting
import os
from analyses.decoding.searchlight.searchlight import searchlight_mode_from_args
from analyses.decoding.searchlight.searchlight_permutation_testing import CHANCE_VALUES, \
    add_searchlight_permutation_args, load_per_subject_scores, permutation_results_dir
from data import TRAINING_MODES
from eval import ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, ACC_CAPTIONS_MOD_SPECIFIC_IMAGES, ACC_IMAGES_MOD_AGNOSTIC, \
    ACC_CAPTIONS_MOD_AGNOSTIC
from utils import RESULTS_DIR, HEMIS, save_plot_and_crop_img, append_images

DEFAULT_VIEWS = ["lateral", "medial", "ventral", "posterior"]
ACC_COLORBAR_MAX = 0.8
COLORBAR_THRESHOLD_MIN = 0.5

COLORBAR_DIFFERENCE_MAX = 0.1

COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.02

CMAP = "cold_hot"
CMAP_POS_ONLY = "hot"

DEFAULT_T_VALUE_THRESH = 1  # 0.824
DEFAULT_TFCE_VAL_THRESH = 10

PLOT_NULL_DISTR_NUM_SAMPLES = 10

METRICS = [ACC_IMAGES_MOD_AGNOSTIC, ACC_IMAGES_MOD_SPECIFIC_CAPTIONS, ACC_CAPTIONS_MOD_AGNOSTIC,
           ACC_CAPTIONS_MOD_SPECIFIC_IMAGES]


def plot_acc_scores(scores, args, results_path, subfolder=""):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    acc_scores_pngs_dir = str(os.path.join(results_path, "acc_scores"))
    if subfolder:
        acc_scores_pngs_dir = os.path.join(acc_scores_pngs_dir, subfolder)
    os.makedirs(acc_scores_pngs_dir, exist_ok=True)

    print(f"plotting acc scores. {subfolder}")
    metrics = scores.metric.unique()
    print("Metrics: ", metrics)

    for metric in metrics:
        threshold = COLORBAR_THRESHOLD_MIN
        if CHANCE_VALUES.get(metric,0.5) == 0:
            threshold = COLORBAR_DIFFERENCE_THRESHOLD_MIN

        score_hemi_metric_avgd = None

        for j, hemi in enumerate(HEMIS):
            for training_mode in TRAINING_MODES:
                score_hemi_metric = scores[
                    (scores.hemi == hemi) & (scores.metric == metric) & (scores.training_mode == training_mode)
                    ].copy()
                score_hemi_metric_avgd = score_hemi_metric.groupby('vertex').aggregate(
                    {'value': 'mean'}).value.values
                print(
                    f"{metric} ({hemi} hemi) mean over subjects: {np.nanmean(score_hemi_metric_avgd):.3f} | max: {np.nanmax(score_hemi_metric.value):.3f}")


                print(
                    f"metric: {metric} {hemi} hemi mean: {np.nanmean(score_hemi_metric_avgd):.2f} | "
                    f"max: {np.nanmax(score_hemi_metric_avgd):.2f}")

                for i, view in enumerate(args.views):
                    plotting.plot_surf_stat_map(
                        fsaverage[f"infl_{hemi}"],
                        score_hemi_metric_avgd,
                        hemi=hemi,
                        view=view,
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        bg_on_data=True,
                        colorbar=False,
                        threshold=threshold,
                        vmax=ACC_COLORBAR_MAX,
                        vmin=0.5 if CHANCE_VALUES[metric] == 0.5 else None,
                        cmap=CMAP_POS_ONLY if CHANCE_VALUES[metric] == 0.5 else CMAP,
                        symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
                    )
                    title = f"{metric}_{view}_{hemi}"
                    save_plot_and_crop_img(os.path.join(acc_scores_pngs_dir, f"{title}.png"))

        if score_hemi_metric_avgd is not None:
            plotting.plot_surf_stat_map(
                fsaverage[f"infl_{HEMIS[0]}"],
                score_hemi_metric_avgd,
                hemi=HEMIS[0],
                view=args.views[0],
                bg_map=fsaverage[f"sulc_{HEMIS[0]}"],
                bg_on_data=True,
                colorbar=True,
                threshold=threshold,
                vmax=ACC_COLORBAR_MAX if CHANCE_VALUES[metric] == 0.5 else COLORBAR_DIFFERENCE_MAX,
                vmin=0.5 if CHANCE_VALUES[metric] == 0.5 else None,
                cmap=CMAP_POS_ONLY if CHANCE_VALUES[metric] == 0.5 else CMAP,
                symmetric_cbar=False if CHANCE_VALUES[metric] == 0.5 else True,
            )
            save_plot_and_crop_img(os.path.join(acc_scores_pngs_dir, f"colorbar_{metric}.png"), crop_cbar=True)


def create_composite_image(args):
    results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
                                    searchlight_mode_from_args(args)))

    tfce_values_img_dir = str(os.path.join(results_path, "tmp", "tfce-values"))
    tfce_val_img = Image.open(os.path.join(tfce_values_img_dir, f"{args.metric}_lateral_left.png"))
    # offset_size = (int(p_val_img.size[0]/10), p_val_img.size[1])
    # image_whitespace = Image.new('RGBA', offset_size, color=(255, 255, 255, 0))
    cbar = Image.open(os.path.join(tfce_values_img_dir, f"colorbar_{args.metric}.png"))
    cbar = cbar.resize((int(cbar.size[0] / 1.2), int(cbar.size[1] / 1.2)))
    tfce_val_img = tfce_val_img.resize((int(tfce_val_img.size[0] * 1.1), int(tfce_val_img.size[1] * 1.1)))
    tfce_val_img = append_images([cbar, tfce_val_img], padding=150)  # image_whitespace

    acc_scores_imgs_dir = str(os.path.join(results_path, "tmp", "acc_scores"))
    acc_scores_imgs = []
    for metric in METRICS:
        acc_scores_img = Image.open(os.path.join(acc_scores_imgs_dir, f"{metric}_lateral_left.png"))
        # if metric in [ACC_IMAGES_MOD_AGNOSTIC, ACC_IMAGES_MOD_SPECIFIC_CAPTIONS]:
        #     acc_scores_img = append_images([cbar, img], padding=50)
        # else:
        #     acc_scores_img = append_images([img, cbar], padding=50)
        acc_scores_img = acc_scores_img.resize((int(acc_scores_img.size[0] / 1.2), int(acc_scores_img.size[1] / 1.2)))
        acc_scores_imgs.append(acc_scores_img)

    # cbar = Image.open(os.path.join(acc_scores_imgs_dir, f"colorbar_{ACC_IMAGES_MOD_AGNOSTIC}.png"))

    acc_scores_imgs_column_1 = append_images(acc_scores_imgs[:2], horizontally=False, padding=400)
    acc_scores_imgs_column_2 = append_images(acc_scores_imgs[2:], horizontally=False, padding=400)

    acc_imgs = append_images([acc_scores_imgs_column_1, acc_scores_imgs_column_2], padding=400)

    full_img = append_images([acc_imgs, tfce_val_img], horizontally=False, padding=300)

    path = os.path.join(results_path, "searchlight_methods.png")
    full_img.save(path, transparent=True)
    print("done")


def run(args):
    results_dir = os.path.join(permutation_results_dir(args), "results")
    os.makedirs(results_dir, exist_ok=True)

    scores = load_per_subject_scores(args)
    plot_acc_scores(scores, args, results_dir)

    # create_composite_image(args)


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    parser.add_argument("--plot-null-distr", default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
