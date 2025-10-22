import argparse

import nibabel.freesurfer
import numpy as np
from PIL import Image
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import os
import pickle

from analyses.decoding.searchlight.searchlight_permutation_testing import calc_significance_cutoff, TFCE_VAL_METRICS, \
    T_VAL_METRICS, DEFAULT_P_VAL_THRESHOLD, T_VAL_METRICS_GW, T_VAL_METRICS_UNATTENDED, T_VAL_METRICS_IMAGERY, \
    T_VAL_METRICS_DECODER_DIFF, T_VAL_METRICS_BASE, T_VAL_METRICS_ATTENTION_DIFF, T_VAL_METRICS_ATTENTION_DIFF_2
from analyses.decoding.searchlight.searchlight import searchlight_mode_from_args
from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, \
    add_searchlight_permutation_args
from analyses.visualization.plotting_utils import plot_surf_contours_custom, plot_surf_stat_map_custom
from utils import RESULTS_DIR, HEMIS, FREESURFER_HOME_DIR, FS_HEMI_NAMES, \
    save_plot_and_crop_img, append_images, METRIC_GW, DIFF, DIFF_DECODERS, METRIC_VISION, METRIC_VISION_2, METRIC_LANG, \
    METRIC_LANG_2

HCP_ATLAS_DIR = os.path.join("atlas_data", "hcp_surface")
HCP_ATLAS_LH = os.path.join(HCP_ATLAS_DIR, "lh.HCP-MMP1.annot")
HCP_ATLAS_RH = os.path.join(HCP_ATLAS_DIR, "rh.HCP-MMP1.annot")

CMAP_POS_ONLY = "hot"
ACC_COLORBAR_MIN = 0.5
ACC_COLORBAR_THRESHOLD = 0.52

CBAR_TFCE_MAX_VALUE = 400000

CONTOUR_COLOR = 'lightseagreen'

DEFAULT_VIEWS = ["lateral", "medial", "ventral"]


def plot(args):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    for result_metric in [METRIC_GW, METRIC_VISION, METRIC_VISION_2, METRIC_LANG, METRIC_LANG_2] + T_VAL_METRICS:
        results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
                                        searchlight_mode_from_args(args)))

        atlas_tmp_results_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))
        os.makedirs(atlas_tmp_results_dir, exist_ok=True)

        args.metric = result_metric

        rois_for_view = {
            METRIC_GW: {
                "left": {
                    "medial": ['precuneus', 'isthmuscingulate', 'parahippocampal'],
                    "lateral": ['inferiorparietal', 'supramarginal', 'middletemporal', 'bankssts'],
                    "ventral": ['inferiortemporal', 'fusiform'],
                },
                "right": {
                    "medial": [],
                    "lateral": [],
                    "ventral": [],
                }
            },
            # ACC_IMAGERY_MOD_AGNOSTIC: {
            #     "left": {
            #         "medial": ['precuneus', 'isthmuscingulate', 'parahippocampal'],
            #         "lateral": ['inferiorparietal', 'supramarginal', 'middletemporal', 'bankssts'],
            #         "ventral": ['inferiortemporal', 'fusiform'],
            #     },
            #     "right": {
            #         "medial": ['precuneus', 'isthmuscingulate', 'parahippocampal'],
            #         "lateral": ['inferiorparietal', 'middletemporal', 'bankssts'],
            #         "ventral": ['inferiortemporal', 'fusiform'],
            #     },
            # }
        }

        result_values = dict()

        if result_metric in [METRIC_GW, METRIC_VISION, METRIC_VISION_2, METRIC_LANG, METRIC_LANG_2]:
            tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values_{result_metric}.p")
            orig_result_values = pickle.load(open(tfce_values_path, "rb"))
            for hemi in HEMIS:
                result_values[hemi] = orig_result_values[hemi][args.metric]
                result_values[hemi] = np.log(result_values[hemi])

            null_distribution_tfce_values_file = os.path.join(
                permutation_results_dir(args),
                f"tfce_values_null_distribution_{result_metric}.p"
            )
            null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))
            significance_cutoff, _ = calc_significance_cutoff(null_distribution_tfce_values, args.metric,
                                                              args.p_value_threshold)
            # significance_cutoff = 17.64
            significance_cutoff = np.log(significance_cutoff)

            print(f"{result_metric} significance cutoff: {significance_cutoff}")
            threshold = significance_cutoff
            cbar_min = 0
            # cbar_max = CBAR_TFCE_MAX_VALUE
            cbar_max = np.nanmax(np.concatenate((result_values['left'], result_values['right'])))
            # print(f"{result_metric} max tfce value across hemis: {cbar_max}")

        elif result_metric.split('$')[0] == DIFF:
            _, training_mode, metric_1, metric_2 = result_metric.split('$')

            for hemi in HEMIS:
                path_mean_acc_values_1 = os.path.join(permutation_results_dir(args), "acc_results_maps",
                                                      f"{training_mode}_decoder_{metric_1}_{FS_HEMI_NAMES[hemi]}.gii")
                path_mean_acc_values_2 = os.path.join(permutation_results_dir(args), "acc_results_maps",
                                                      f"{training_mode}_decoder_{metric_2}_{FS_HEMI_NAMES[hemi]}.gii")
                result_values[hemi] = nibabel.load(path_mean_acc_values_1).darrays[0].data - \
                                      nibabel.load(path_mean_acc_values_2).darrays[0].data

            threshold = 0.01
            cbar_min = 0.01
            cbar_max = 0.2  # np.nanmax(np.concatenate((result_values['left'], result_values['right'])))
        elif result_metric.split('$')[0] == DIFF_DECODERS:
            _, training_mode_1, training_mode_2, metric_name = result_metric.split('$')

            for hemi in HEMIS:
                path_mean_acc_values_1 = os.path.join(permutation_results_dir(args), "acc_results_maps",
                                                      f"{training_mode_1}_decoder_{metric_name}_{FS_HEMI_NAMES[hemi]}.gii")
                path_mean_acc_values_2 = os.path.join(permutation_results_dir(args), "acc_results_maps",
                                                      f"{training_mode_2}_decoder_{metric_name}_{FS_HEMI_NAMES[hemi]}.gii")
                result_values[hemi] = nibabel.load(path_mean_acc_values_1).darrays[0].data - \
                                      nibabel.load(path_mean_acc_values_2).darrays[0].data

            threshold = 0.01
            cbar_min = 0.01
            cbar_max = 0.2  # np.nanmax(np.concatenate((result_values['left'], result_values['right'])))
        else:
            training_mode, metric = result_metric.split('$')

            for hemi in HEMIS:
                path_mean_acc_values = os.path.join(permutation_results_dir(args), "acc_results_maps",
                                                    f"{training_mode}_decoder_{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                result_values[hemi] = nibabel.load(path_mean_acc_values).darrays[0].data

            threshold = 0.5
            cbar_min = 0.5
            cbar_max = 0.75  # np.nanmax(np.concatenate((result_values['left'], result_values['right'])))

        print(f"{result_metric} cbar max: {cbar_max}")
        for hemi in HEMIS:
            hemi_fs = FS_HEMI_NAMES[hemi]
            # atlas_path = os.path.join(FREESURFER_HOME_DIR, f"subjects/fsaverage/label/{hemi_fs}.aparc.a2009s.annot")
            atlas_path = os.path.join(FREESURFER_HOME_DIR, f"subjects/fsaverage/label/{hemi_fs}.aparc.annot")
            atlas_labels, atlas_colors, names = nibabel.freesurfer.read_annot(atlas_path)
            names = [name.decode() for name in names]

            if result_metric in rois_for_view:
                for i, (view, rois) in enumerate(rois_for_view[result_metric][hemi].items()):
                    regions_indices = [names.index(roi) for roi in rois]
                    # label_names = [label_names_dict[r] if r in label_names_dict else r.replace("-", " ") for r in rois]
                    label_names = [r for r in rois]
                    # colors = [all_colors[label_names[regions_indices.index(i)]] if i in regions_indices else (0, 0, 0, 0) for i
                    #           in range(np.nanmax(atlas_labels))]
                    # cmap = ListedColormap(colors)
                    atlas_labels_current_view = np.array([l if l in regions_indices else np.nan for l in atlas_labels])
                    fig = plotting.plot_surf_stat_map(
                        fsaverage[f"infl_{hemi}"],
                        result_values[hemi],
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        hemi=hemi,
                        view=view,
                        colorbar=False,
                        threshold=threshold,
                        vmax=cbar_max,
                        vmin=cbar_min,
                        cmap=CMAP_POS_ONLY,
                    )
                    plot_surf_contours_custom(
                        surf_mesh=fsaverage[f"infl_{hemi}"],
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        roi_map=atlas_labels_current_view,
                        levels=regions_indices,
                        hemi=hemi,
                        figure=fig,
                        colors=[CONTOUR_COLOR] * len(regions_indices),
                    )

                    title = f"{view}_{hemi}"
                    path = os.path.join(atlas_tmp_results_dir, f"{title}.png")
                    save_plot_and_crop_img(path)
                    print(f"saved {path}")
            else:
                for view in args.views:
                    fig = plotting.plot_surf_stat_map(
                        fsaverage[f"infl_{hemi}"],
                        result_values[hemi],
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        hemi=hemi,
                        view=view,
                        colorbar=False,
                        threshold=threshold,
                        vmax=cbar_max,
                        vmin=cbar_min,
                        cmap=CMAP_POS_ONLY,
                    )

                    title = f"{view}_{hemi}"
                    path = os.path.join(atlas_tmp_results_dir, f"{title}.png")
                    save_plot_and_crop_img(path)
                    print(f"saved {path}")

        # plot for cbar:
        fig = plt.figure(figsize=(7, 6))
        plot_surf_stat_map_custom(
            fsaverage[f"infl_{HEMIS[0]}"],
            result_values[HEMIS[0]],
            hemi=HEMIS[0],
            view=args.views[0],
            colorbar=True,
            threshold=threshold,
            vmax=cbar_max,
            vmin=cbar_min,
            cmap=CMAP_POS_ONLY,
            figure=fig,
            metric=result_metric,
        )
        save_plot_and_crop_img(os.path.join(atlas_tmp_results_dir, "colorbar.png"), crop_cbar=True,
                               horizontal_cbar=True)


def create_composite_images_of_all_views(args):
    for result_metric in [METRIC_GW, METRIC_VISION, METRIC_VISION_2, METRIC_LANG, METRIC_LANG_2] + T_VAL_METRICS:
        results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
                                        searchlight_mode_from_args(args)))

        results_values_imgs_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))

        images_lateral = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in
                          ["lateral"] for hemi in HEMIS]
        images_medial = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["medial"]
                         for hemi in HEMIS]
        # images_posterior = [Image.open(os.path.join(tfce_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["posterior"] for hemi
        #                  in HEMIS]

        imgs_ventral = [Image.open(os.path.join(results_values_imgs_dir, f"ventral_{hemi}.png")) for hemi in HEMIS]
        img_ventral = append_images(images=imgs_ventral, horizontally=False)

        img_medial = append_images(images=images_medial, padding=20)
        # img_posterior = append_images(images=images_posterior)

        img_lateral = append_images(images=images_lateral, padding=20)

        img_colorbar = Image.open(os.path.join(results_values_imgs_dir, "colorbar.png"))
        offset_size = (img_colorbar.size[0], int(img_lateral.size[1] - img_colorbar.size[1]))
        image_whitespace = Image.new('RGBA', offset_size, color=(255, 255, 255, 0))
        img_colorbar = append_images([image_whitespace, img_colorbar], horizontally=False)

        img_row_1 = append_images([img_lateral], padding=10)
        img_row_2 = append_images([img_medial], padding=10)
        img_row_3 = append_images([img_ventral, img_colorbar], padding=30)

        # roi_legend = Image.open(os.path.join(tfce_values_imgs_dir, f"legend.png"))

        composite_image = append_images([img_row_1, img_row_2, img_row_3], padding=5, horizontally=True)

        path = os.path.join(results_path, "searchlight_results",
                            f"{result_metric.replace('_', '-').replace('$', '_')}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        composite_image.save(path, transparent=True)  # , facecolor="black")
        print('saved ', path)


def create_composite_images_of_metrics(args):
    for name, metrics in zip(
            ['metrics_base', 'metrics_attention_diff', 'metrics_attention_diff_2', 'metrics_unattended_stimuli',
             'metrics_decoder_diffs',
             'metrics_imagery'],
            [T_VAL_METRICS_BASE, T_VAL_METRICS_ATTENTION_DIFF, T_VAL_METRICS_ATTENTION_DIFF_2, T_VAL_METRICS_UNATTENDED,
             T_VAL_METRICS_DECODER_DIFF,
             T_VAL_METRICS_IMAGERY]):
        imgs = []
        for result_metric in metrics:
            results_path = os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
                                        searchlight_mode_from_args(args), "searchlight_results",
                                        f"{result_metric.replace('_', '-').replace('$', '_')}.png")
            imgs.append(Image.open(results_path))
        img_all_metrics = append_images(images=imgs, padding=20, horizontally=False)
        img_all_metrics_path = os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
                                            searchlight_mode_from_args(args), "searchlight_results", f"{name}.png")

        # Add background color
        background = Image.new('RGBA', img_all_metrics.size, (255, 255, 255))
        with_bg = Image.alpha_composite(background, img_all_metrics)

        with_bg.save(img_all_metrics_path)


# def create_composite_image(args):
#     for result_metric in T_VAL_METRICS + [METRIC_MOD_INVARIANT]:
#         results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
#                                         searchlight_mode_from_args(args)))
#
#         results_values_imgs_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))
#
#         images_lateral = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in
#                           ["lateral"] for hemi
#                           in HEMIS]
#         images_medial = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["medial"]
#                          for hemi
#                          in HEMIS]
#         # images_posterior = [Image.open(os.path.join(tfce_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["posterior"] for hemi
#         #                  in HEMIS]
#
#         imgs_ventral = [Image.open(os.path.join(results_values_imgs_dir, f"ventral_{hemi}.png")) for hemi in HEMIS]
#         img_ventral = append_images(images=imgs_ventral, horizontally=False)
#
#         img_medial = append_images(images=images_medial, padding=400)
#         # img_posterior = append_images(images=images_posterior)
#
#         img_lateral = append_images(images=images_lateral, padding=400)
#
#         img_colorbar = Image.open(os.path.join(results_values_imgs_dir, "colorbar.png"))
#         offset_size = (img_colorbar.size[0], int(img_lateral.size[1] - img_colorbar.size[1]))
#         image_whitespace = Image.new('RGBA', offset_size, color=(255, 255, 255, 0))
#         img_colorbar = append_images([image_whitespace, img_colorbar], horizontally=False)
#
#         img_row_1 = append_images([img_lateral], padding=50)
#         img_row_2 = append_images([img_medial], padding=30)
#         img_row_3 = append_images([img_ventral, img_colorbar], padding=300)
#
#         # roi_legend = Image.open(os.path.join(tfce_values_imgs_dir, f"legend.png"))
#
#         # p_val_image = append_images([img_row_1, img_row_2, roi_legend], padding=5, horizontally=False)
#         composite_image = append_images([img_row_1, img_row_2, img_row_3], padding=5, horizontally=False)
#
#         path = os.path.join(results_path, "searchlight_results", f"{result_metric}.png")
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         composite_image.save(path, transparent=True)  # , facecolor="black")


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)
    parser.add_argument("--p-value-threshold", type=float, default=DEFAULT_P_VAL_THRESHOLD)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    plot(args)
    create_composite_images_of_all_views(args)
    create_composite_images_of_metrics(args)
