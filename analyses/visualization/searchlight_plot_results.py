import argparse

import nibabel.freesurfer
import numpy as np
from PIL import Image
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import os
import pickle

from analyses.decoding.searchlight.searchlight_permutation_testing import calc_significance_cutoff, TFCE_VAL_METRICS, \
    T_VAL_METRICS, DEFAULT_P_VAL_THRESHOLD
from analyses.decoding.searchlight.searchlight import searchlight_mode_from_args
from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, \
    add_searchlight_permutation_args
from analyses.visualization.plotting_utils import plot_surf_contours_custom, plot_surf_stat_map_custom
from utils import RESULTS_DIR, HEMIS, FREESURFER_HOME_DIR, FS_HEMI_NAMES, \
    save_plot_and_crop_img, append_images, METRIC_MOD_INVARIANT, DIFF

HCP_ATLAS_DIR = os.path.join("atlas_data", "hcp_surface")
HCP_ATLAS_LH = os.path.join(HCP_ATLAS_DIR, "lh.HCP-MMP1.annot")
HCP_ATLAS_RH = os.path.join(HCP_ATLAS_DIR, "rh.HCP-MMP1.annot")

CMAP_POS_ONLY = "hot"
ACC_COLORBAR_MIN = 0.5
ACC_COLORBAR_THRESHOLD = 0.52

CBAR_TFCE_MAX_VALUE = 400000

CONTOUR_COLOR = 'lightseagreen'

DEFAULT_VIEWS = ["lateral", "medial", "ventral", "posterior"]


def plot(args):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    for result_metric in T_VAL_METRICS + [METRIC_MOD_INVARIANT]:
        results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
                                        searchlight_mode_from_args(args)))

        atlas_tmp_results_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))
        os.makedirs(atlas_tmp_results_dir, exist_ok=True)

        args.metric = result_metric

        rois_for_view = {
            METRIC_MOD_INVARIANT: {
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

        if result_metric in [METRIC_MOD_INVARIANT]:
            tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values_{result_metric}.p")
            orig_result_values = pickle.load(open(tfce_values_path, "rb"))
            for hemi in HEMIS:
                result_values[hemi] = orig_result_values[hemi][args.metric]

            #TODO
            # null_distribution_tfce_values_file = os.path.join(
            #     permutation_results_dir(args),
            #     f"tfce_values_null_distribution_{result_metric}.p"
            # )
            # null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))
            # significance_cutoff, _ = calc_significance_cutoff(null_distribution_tfce_values, args.metric,
            #                                                   args.p_value_threshold)
            significance_cutoff = 6.72
            print(f"{result_metric} significance cutoff: {significance_cutoff}")
            threshold = significance_cutoff
            cbar_min = 0
            # cbar_max = CBAR_TFCE_MAX_VALUE
            cbar_max = np.nanmax(np.concatenate((result_values['left'], result_values['right'])))
            # print(f"{result_metric} max tfce value across hemis: {cbar_max}")

        # elif result_metric.startswith("pairwise_acc"):
        #     # cbar_min = ACC_COLORBAR_MIN
        #     # cbar_max = COLORBAR_MAX
        #     # threshold = ACC_COLORBAR_THRESHOLD
        #     # subject_scores = load_per_subject_scores(args)
        #     # for hemi in HEMIS:
        #     #     score_hemi_avgd = np.nanmean([subject_scores[subj][hemi][result_metric] for subj in args.subjects],
        #     #                                  axis=0)
        #     #     result_values[hemi] = score_hemi_avgd
        #
        #     # t_values = pickle.load(open(os.path.join(permutation_results_dir(args), "t_values.p"), 'rb'))
        #     # for hemi in HEMIS:
        #     #     result_values[hemi] = t_values[hemi][args.metric]
        #     tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values_{result_metric}.p")
        #     orig_result_values = pickle.load(open(tfce_values_path, "rb"))
        #     for hemi in HEMIS:
        #         result_values[hemi] = orig_result_values[hemi][args.metric]
        #
        #     null_distribution_tfce_values_file = os.path.join(
        #         permutation_results_dir(args),
        #         f"tfce_values_null_distribution_{result_metric}.p"
        #     )
        #     null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))
        #     significance_cutoff, _ = calc_significance_cutoff(null_distribution_tfce_values, args.metric,
        #                                                       args.p_value_threshold)
        #     threshold = significance_cutoff
        #     cbar_min = 0
        #     cbar_max = np.nanmax(np.concatenate((result_values['left'], result_values['right'])))
        #     # print(f"{result_metric} max tfce value across hemis: {np.nanmax(np.concatenate((result_values['left'], result_values['right'])))}")
        #
        #     # for hemi in HEMIS:
        #     #     print(f"{hemi} hemi fraction of values above thresh: {np.mean(result_values[hemi] > significance_cutoffs[hemi])}")
        #     #     result_values[hemi][result_values[hemi] < significance_cutoffs[hemi]] = 0
        #     #     print([round(val) for val in result_values[hemi][result_values[hemi] > 2000][:20]])
        #     # threshold = 1
        #
        #     # from t-val table:
        #     # for p<0.05: 2.015
        #     # for p<0.01: 3.365
        #     # for p<0.001: 5.893
        #
        #     # from permutation testing:
        #     # test statistic significance cutoff for p<0.05: 2.06
        #     # min mean acc:
        #     # 0.5308641975308642
        #     # test statistic significance cutoff for p<0.01: 3.44
        #     # min mean acc:
        #     # 0.5740740740740741
        #     # test statistic significance cutoff for p<0.001: 6.03
        #     # min mean acc: 0.5902777777777778
        #     # threshold = 2.015
        #     # cbar_min = 0
        #     # cbar_max = CBAR_T_VAL_MAX

        elif result_metric.startswith(DIFF):
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


def create_composite_image(args):
    for result_metric in T_VAL_METRICS + [METRIC_MOD_INVARIANT]:
        results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
                                        searchlight_mode_from_args(args)))

        results_values_imgs_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))

        images_lateral = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in
                          ["lateral"] for hemi
                          in HEMIS]
        images_medial = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["medial"]
                         for hemi
                         in HEMIS]
        # images_posterior = [Image.open(os.path.join(tfce_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["posterior"] for hemi
        #                  in HEMIS]

        imgs_ventral = [Image.open(os.path.join(results_values_imgs_dir, f"ventral_{hemi}.png")) for hemi in HEMIS]
        img_ventral = append_images(images=imgs_ventral, horizontally=False)

        img_medial = append_images(images=images_medial, padding=400)
        # img_posterior = append_images(images=images_posterior)

        img_lateral = append_images(images=images_lateral, padding=400)

        img_colorbar = Image.open(os.path.join(results_values_imgs_dir, "colorbar.png"))
        offset_size = (img_colorbar.size[0], int(img_lateral.size[1] - img_colorbar.size[1]))
        image_whitespace = Image.new('RGBA', offset_size, color=(255, 255, 255, 0))
        img_colorbar = append_images([image_whitespace, img_colorbar], horizontally=False)

        img_row_1 = append_images([img_lateral], padding=50)
        img_row_2 = append_images([img_medial], padding=30)
        img_row_3 = append_images([img_ventral, img_colorbar], padding=300)

        # roi_legend = Image.open(os.path.join(tfce_values_imgs_dir, f"legend.png"))

        # p_val_image = append_images([img_row_1, img_row_2, roi_legend], padding=5, horizontally=False)
        p_val_image = append_images([img_row_1, img_row_2, img_row_3], padding=5, horizontally=False)

        path = os.path.join(results_path, f"searchlight_results_{result_metric}.png")
        p_val_image.save(path, transparent=True)  # , facecolor="black")


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)
    parser.add_argument("--p-value-threshold", type=float, default=DEFAULT_P_VAL_THRESHOLD)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    plot(args)
    create_composite_image(args)
