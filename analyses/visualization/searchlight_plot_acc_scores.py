import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.figure import Figure
from nilearn import datasets, plotting
import os
from analyses.decoding.searchlight.searchlight_permutation_testing import CHANCE_VALUES, \
    add_searchlight_permutation_args, load_per_subject_scores, permutation_results_dir
from data import TRAINING_MODES, MODALITY_AGNOSTIC, TEST_SPLITS
from utils import HEMIS, save_plot_and_crop_img, append_images

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


def plot_acc_scores(scores, args, results_path, subfolder=""):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    acc_scores_pngs_dir = str(os.path.join(results_path, "acc_scores"))
    if subfolder:
        acc_scores_pngs_dir = os.path.join(acc_scores_pngs_dir, subfolder)
    os.makedirs(acc_scores_pngs_dir, exist_ok=True)

    print(f"plotting acc scores. {subfolder}")

    for metric in TEST_SPLITS:
        threshold = COLORBAR_THRESHOLD_MIN
        chance_value = CHANCE_VALUES.get(metric, 0.5)
        print(f"{metric} | chance value: {chance_value}")
        if chance_value == 0:
            threshold = COLORBAR_DIFFERENCE_THRESHOLD_MIN

        score_hemi_metric_avgd = None

        for j, hemi in enumerate(HEMIS):
            training_mode = MODALITY_AGNOSTIC
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
                    vmin=0.5 if chance_value == 0.5 else None,
                    cmap=CMAP_POS_ONLY if chance_value == 0.5 else CMAP,
                    symmetric_cbar=False if chance_value == 0.5 else True,
                )
                title = f"{training_mode}_decoder_{metric}_{view}_{hemi}"
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
                vmax=ACC_COLORBAR_MAX if chance_value == 0.5 else COLORBAR_DIFFERENCE_MAX,
                vmin=0.5 if chance_value == 0.5 else None,
                cmap=CMAP_POS_ONLY if chance_value == 0.5 else CMAP,
                symmetric_cbar=False if chance_value == 0.5 else True,
            )
            save_plot_and_crop_img(os.path.join(acc_scores_pngs_dir, f"colorbar_{metric}.png"), crop_cbar=True)


def create_composite_image(args, results_path):
    acc_scores_pngs_dir = str(os.path.join(results_path, "acc_scores"))

    training_mode = MODALITY_AGNOSTIC

    imgs_metrics = []
    for metric in TEST_SPLITS:
        imgs_views = []
        for view in args.views:
            imgs_hemis = []
            for hemi in HEMIS:
                imgs_hemis.append(Image.open(os.path.join(acc_scores_pngs_dir, f"{training_mode}_decoder_{metric}_{view}_{hemi}.png")))
            img_hemi = append_images(images=imgs_hemis, padding=10, horizontally=False if view == 'ventral' else True)
            imgs_views.append(img_hemi)

        # title_img = Image.new('RGB', (200, imgs_views[0].size[1]), color='white')
        # draw = ImageDraw.Draw(title_img)
        # font = ImageFont.load_default(size=24)
        # draw.text((0, 100), metric, (0, 0, 0), font=font)

        fig = Figure(facecolor="none")
        fig.text(0, 0, metric, fontsize=50)
        fig.savefig(results_path+'tmptitle.png')
        title_img = Image.open(results_path+'tmptitle.png')

        imgs_views = [title_img] + imgs_views
        img_views = append_images(images=imgs_views, padding=200)
        imgs_metrics.append(img_views)

        cbar = Image.open(os.path.join(acc_scores_pngs_dir, f"colorbar_{metric}.png"))
        img_views = append_images(images=[img_views, cbar], padding=200)

        path = os.path.join(results_path, f"{training_mode}_{metric}.png")
        img_views.save(path, transparent=True)
        print(f'saved {path}')

    imgs_metrics = append_images(images=imgs_metrics, padding=50, horizontally=False)
    path = os.path.join(results_path, f"{training_mode}.png")
    imgs_metrics.save(path, transparent=True)
    print("done")


def run(args):
    results_dir = os.path.join(permutation_results_dir(args), "results")
    os.makedirs(results_dir, exist_ok=True)

    # scores = load_per_subject_scores(args)
    # plot_acc_scores(scores, args, results_dir)

    create_composite_image(args, results_dir)


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
