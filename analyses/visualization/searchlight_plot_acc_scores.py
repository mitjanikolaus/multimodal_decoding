import argparse

import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from nilearn import datasets, plotting
import os
from analyses.decoding.searchlight.searchlight_permutation_testing import add_searchlight_permutation_args, load_per_subject_scores, permutation_results_dir
from analyses.visualization.plotting_utils import add_hemi_label
from data import TRAINING_MODES, MODALITY_AGNOSTIC, TEST_SPLITS, TEST_IMAGES_ATTENDED, \
    TEST_IMAGES_UNATTENDED, TEST_CAPTIONS_ATTENDED, TEST_CAPTIONS_UNATTENDED, \
    MODALITY_SPECIFIC_IMAGES, MODALITY_SPECIFIC_CAPTIONS, SPLIT_IMAGERY_WEAK
from utils import HEMIS, save_plot_and_crop_img, append_images, FS_NUM_VERTICES, DIFF, DIFF_DECODERS

DEFAULT_VIEWS = ["lateral", "medial", "ventral"]
ACC_COLORBAR_MAX = 0.7
ACC_COLORBAR_MAX_IMAGERY = 0.8

COLORBAR_THRESHOLD_MIN = 0.53
COLORBAR_THRESHOLD_MIN_IMAGERY = 0.5

COLORBAR_DIFFERENCE_MAX = 0.1

COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.05

CMAP = "cold_hot"
CMAP_POS_ONLY = "hot"
CMAP_POS_ONLY_IMAGERY = "magma"

DEFAULT_T_VALUE_THRESH = 1  # 0.824
DEFAULT_TFCE_VAL_THRESH = 10

PLOT_NULL_DISTR_NUM_SAMPLES = 10

DIFF_METRICS = [
    '$'.join([DIFF, TEST_IMAGES_ATTENDED, TEST_IMAGES_UNATTENDED]),
    '$'.join([DIFF, TEST_CAPTIONS_ATTENDED, TEST_CAPTIONS_UNATTENDED]),
]


def plot_acc_scores(scores, args, results_path, subfolder="", training_mode=MODALITY_AGNOSTIC,
                    make_per_subject_plots=True):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    acc_scores_pngs_dir = str(os.path.join(results_path, "acc_scores"))
    if subfolder:
        acc_scores_pngs_dir = os.path.join(acc_scores_pngs_dir, subfolder)
    os.makedirs(acc_scores_pngs_dir, exist_ok=True)

    print(f"plotting acc scores. {subfolder}")

    for metric in DIFF_METRICS + TEST_SPLITS:
        threshold = COLORBAR_THRESHOLD_MIN_IMAGERY if "imagery" in metric else COLORBAR_THRESHOLD_MIN
        chance_value = 0 if metric.split('$')[0] in [DIFF, DIFF_DECODERS] else 0.5
        print(f"{metric} | chance value: {chance_value}")
        if chance_value == 0:
            threshold = COLORBAR_DIFFERENCE_THRESHOLD_MIN

        if chance_value == 0.5:
            acc_colorbar_max = ACC_COLORBAR_MAX_IMAGERY if "imagery" in metric else ACC_COLORBAR_MAX
        else:
            acc_colorbar_max = COLORBAR_DIFFERENCE_MAX

        score_hemi_metric_avgd = None

        for hemi in HEMIS:
            if metric.split('$')[0] == DIFF:
                metric_name_1, metric_name_2 = metric.split('$')[1:]
                scores_filtered = scores[(scores.hemi == hemi) & (scores.training_mode == training_mode)]
                score_hemi_metric_1_avgd = scores_filtered[scores_filtered.metric == metric_name_1].groupby('vertex').aggregate(
                    {'value': 'mean'}).value.values
                score_hemi_metric_2_avgd = scores_filtered[scores_filtered.metric == metric_name_2].groupby('vertex').aggregate(
                    {'value': 'mean'}).value.values
                score_hemi_metric_avgd = score_hemi_metric_1_avgd - score_hemi_metric_2_avgd
            else:
                score_hemi_metric = scores[
                    (scores.hemi == hemi) & (scores.metric == metric) & (scores.training_mode == training_mode)
                    ].copy()
                score_hemi_metric_avgd = score_hemi_metric.groupby('vertex').aggregate(
                    {'value': 'mean'}).value.values
            print(
                f"metric: {metric} {hemi} hemi mean over subjects: {np.nanmean(score_hemi_metric_avgd):.2f} | "
                f"max: {np.nanmax(score_hemi_metric_avgd):.2f}"
            )

            for i, view in enumerate(args.views):
                fig = plotting.plot_surf_stat_map(
                    fsaverage[f"infl_{hemi}"],
                    score_hemi_metric_avgd,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    colorbar=False,
                    threshold=threshold,
                    vmax=acc_colorbar_max,
                    vmin=0.5 if chance_value == 0.5 else None,
                    cmap=CMAP_POS_ONLY_IMAGERY if "imagery" in metric else CMAP_POS_ONLY,# if chance_value == 0.5 else CMAP,
                    symmetric_cbar=False,# if chance_value == 0.5 else True,
                )
                add_hemi_label(fig, hemi, view)
                title = f"{training_mode}_decoder_{metric}_{view}_{hemi}"
                save_plot_and_crop_img(os.path.join(acc_scores_pngs_dir, f"{title}.png"))
                print(f'saved {os.path.join(acc_scores_pngs_dir, f"{title}.png")}')

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
                vmax=acc_colorbar_max,
                vmin=0.5 if chance_value == 0.5 else None,
                cmap=CMAP_POS_ONLY_IMAGERY if "imagery" in metric else CMAP_POS_ONLY,
                # if chance_value == 0.5 else CMAP,
                symmetric_cbar=False,# if chance_value == 0.5 else True,
            )
            save_plot_and_crop_img(os.path.join(acc_scores_pngs_dir, f"colorbar_{metric}.png"), crop_cbar=True,
                               horizontal_cbar=False, crop_to_content=True)

        if make_per_subject_plots:
            for subject in args.subjects:
                for hemi in HEMIS:
                    if metric.split('$')[0] == DIFF:
                        metric_name_1, metric_name_2 = metric.split('$')[1:]
                        scores_filtered = scores[(scores.hemi == hemi) & (scores.training_mode == training_mode) & (
                                        scores.subject == subject)]
                        score_hemi_metric_1 = scores_filtered[scores_filtered.metric == metric_name_1].groupby(
                            'vertex').aggregate(
                            {'value': 'mean'}).value.values
                        score_hemi_metric_2 = scores_filtered[scores_filtered.metric == metric_name_2].groupby(
                            'vertex').aggregate(
                            {'value': 'mean'}).value.values
                        score_hemi_metric = score_hemi_metric_1 - score_hemi_metric_2
                    else:
                        score_hemi_metric = scores[
                            (scores.hemi == hemi) & (scores.metric == metric) & (scores.training_mode == training_mode) & (
                                        scores.subject == subject)
                            ].copy()
                        score_hemi_metric = score_hemi_metric.value.values
                    print(
                        f"{metric} ({hemi} hemi) mean for {subject}: {np.nanmean(score_hemi_metric):.3f} | max: {np.nanmax(score_hemi_metric):.3f}")
                    assert len(score_hemi_metric) == FS_NUM_VERTICES, score_hemi_metric

                    for i, view in enumerate(args.views):
                        fig = plotting.plot_surf_stat_map(
                            fsaverage[f"infl_{hemi}"],
                            score_hemi_metric,
                            hemi=hemi,
                            view=view,
                            bg_map=fsaverage[f"sulc_{hemi}"],
                            bg_on_data=True,
                            colorbar=False,
                            threshold=threshold,
                            vmax=acc_colorbar_max,
                            vmin=0.5 if chance_value == 0.5 else None,
                            cmap=CMAP_POS_ONLY_IMAGERY if "imagery" in metric else CMAP_POS_ONLY,
                            # if chance_value == 0.5 else CMAP,
                            symmetric_cbar=False,# if chance_value == 0.5 else True,
                        )
                        add_hemi_label(fig, hemi, view)
                        title = f"{training_mode}_decoder_{metric}_{view}_{hemi}"
                        out_path = os.path.join(acc_scores_pngs_dir, subject, f"{title}.png")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        save_plot_and_crop_img(out_path)
                        print(f'saved {out_path}')


def create_composite_image(args, results_path, metrics=TEST_SPLITS, training_mode=MODALITY_AGNOSTIC, file_suffix="", make_per_subject_plots=True):
    acc_scores_pngs_dir = str(os.path.join(results_path, "acc_scores"))

    imgs_metrics = []
    for metric in metrics:
        imgs_views = []
        for view in args.views:
            imgs_hemis = []
            for hemi in HEMIS:
                imgs_hemis.append(Image.open(
                    os.path.join(acc_scores_pngs_dir, f"{training_mode}_decoder_{metric}_{view}_{hemi}.png")))
            img_hemi = append_images(images=imgs_hemis, padding=10, horizontally=False if view == 'ventral' else True)
            imgs_views.append(img_hemi)

        fig = Figure(facecolor="none", figsize=(14, 6))
        fig.text(0, 0.9, metric, fontsize=50, fontweight='bold')
        fig.savefig(results_path + 'tmptitle.png')
        title_img = Image.open(results_path + 'tmptitle.png')
        os.remove(results_path + 'tmptitle.png')

        cbar = Image.open(os.path.join(acc_scores_pngs_dir, f"colorbar_{metric}.png"))

        # imgs_views = [title_img] + imgs_views + [cbar]
        imgs_views = imgs_views + [cbar]

        img_views = append_images(images=imgs_views, padding=200)
        imgs_metrics.append(img_views)

        path = os.path.join(results_path, f"{training_mode}_{metric}.png")
        img_views.save(path, transparent=True)
        print(f'saved {path}')

    imgs_metrics = append_images(images=imgs_metrics, padding=50, horizontally=False)
    path = os.path.join(results_path, f"{training_mode}{file_suffix}.png")
    imgs_metrics.save(path, transparent=True)

    if make_per_subject_plots:
        for subject in args.subjects:
            os.makedirs(os.path.join(results_path, subject), exist_ok=True)

            imgs_metrics = []
            for metric in metrics:
                imgs_views = []
                for view in args.views:
                    imgs_hemis = []
                    for hemi in HEMIS:
                        imgs_hemis.append(Image.open(
                            os.path.join(acc_scores_pngs_dir, subject, f"{training_mode}_decoder_{metric}_{view}_{hemi}.png")))
                    img_hemi = append_images(images=imgs_hemis, padding=10,
                                             horizontally=False if view == 'ventral' else True)
                    imgs_views.append(img_hemi)

                fig = Figure(facecolor="none", figsize=(10, 6))
                fig.text(0, 0.9, metric, fontsize=50, fontweight='bold')
                fig.savefig(results_path + 'tmptitle.png')
                title_img = Image.open(results_path + 'tmptitle.png')
                os.remove(results_path + 'tmptitle.png')

                cbar = Image.open(os.path.join(acc_scores_pngs_dir, f"colorbar_{metric}.png"))

                # imgs_views = [title_img] + imgs_views + [cbar]
                imgs_views = imgs_views + [cbar]
                img_views = append_images(images=imgs_views, padding=200)
                imgs_metrics.append(img_views)

                path = os.path.join(results_path, subject, f"{training_mode}_{metric}.png")
                img_views.save(path, transparent=True)
                print(f'saved {path}')

            imgs_metrics = append_images(images=imgs_metrics, padding=50, horizontally=False)
            path = os.path.join(results_path, subject, f"{training_mode}{file_suffix}.png")
            imgs_metrics.save(path, transparent=True)
            print(f'saved {path}')


def run(args):
    results_dir = os.path.join(permutation_results_dir(args), "results")
    os.makedirs(results_dir, exist_ok=True)

    scores = load_per_subject_scores(args)
    # scores = add_diff_metrics(scores)

    make_per_subject_plots = False
    for training_mode in [MODALITY_SPECIFIC_IMAGES, MODALITY_SPECIFIC_CAPTIONS, MODALITY_AGNOSTIC]:
        plot_acc_scores(scores, args, results_dir, training_mode=training_mode,
                        make_per_subject_plots=make_per_subject_plots)

        create_composite_image(args, results_dir, metrics=DIFF_METRICS+[TEST_IMAGES_ATTENDED, TEST_IMAGES_UNATTENDED,
                                                           TEST_CAPTIONS_ATTENDED,
                                                           TEST_CAPTIONS_UNATTENDED],
                               file_suffix="_attention_mod", training_mode=training_mode,
                               make_per_subject_plots=make_per_subject_plots)

        create_composite_image(args, results_dir, training_mode=training_mode, metrics=[SPLIT_IMAGERY_WEAK],
                               make_per_subject_plots=make_per_subject_plots)


    # metrics = [TEST_CAPTIONS_ATTENDED, TEST_CAPTIONS_UNATTENDED, 'diff_attended_unattended_captions']
    # create_composite_image(args, results_dir, metrics=metrics,
    #                        file_suffix="_cross_decoding", training_mode=MODALITY_SPECIFIC_IMAGES)
    #
    # metrics = [TEST_IMAGES_ATTENDED, TEST_IMAGES_UNATTENDED, 'diff_attended_unattended_images']
    # create_composite_image(args, results_dir, metrics=metrics,
    #                        file_suffix="_cross_decoding", training_mode=MODALITY_SPECIFIC_CAPTIONS)

    print("done")


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
