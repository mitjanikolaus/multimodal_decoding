import argparse

import nibabel.freesurfer
import numpy as np
from PIL import Image
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import os
import pickle

from analyses.cluster_analysis import calc_significance_cutoff
from analyses.decoding.searchlight.searchlight import searchlight_mode_from_args
from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, \
    get_hparam_suffix, add_searchlight_permutation_args
from analyses.visualization.plotting_utils import plot_surf_contours_custom, plot_surf_stat_map_custom
from analyses.visualization.searchlight_plot_method import DEFAULT_VIEWS
from eval import ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, ACC_IMAGERY_MOD_AGNOSTIC
from utils import RESULTS_DIR, HEMIS, FREESURFER_HOME_DIR, FS_HEMI_NAMES, METRIC_MOD_AGNOSTIC_AND_CROSS, \
    save_plot_and_crop_img, append_images, METRIC_CROSS_DECODING

HCP_ATLAS_DIR = os.path.join("atlas_data", "hcp_surface")
HCP_ATLAS_LH = os.path.join(HCP_ATLAS_DIR, "lh.HCP-MMP1.annot")
HCP_ATLAS_RH = os.path.join(HCP_ATLAS_DIR, "rh.HCP-MMP1.annot")

CMAP_POS_ONLY = "hot"
ACC_COLORBAR_MIN = 0.5
ACC_COLORBAR_THRESHOLD = 0.52

CBAR_TFCE_MAX_VALUE = 400000

METRICS = [METRIC_CROSS_DECODING, METRIC_MOD_AGNOSTIC_AND_CROSS, ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC]

CONTOUR_COLOR = 'lightseagreen'


def plot(args):
    # plt.style.use("dark_background")

    for result_metric in METRICS:
        results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution,
                                        searchlight_mode_from_args(args)))

        atlas_tmp_results_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))
        os.makedirs(atlas_tmp_results_dir, exist_ok=True)

        args.metric = result_metric

        fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

        # rois_for_view = {
        #     "medial": ['G_precuneus', 'S_subparietal', 'G_cingul-Post-dorsal', 'S_parieto_occipital',
        #                'G_oc-temp_med-Parahip', 'S_pericallosal'],
        #     "lateral": ['G_pariet_inf-Angular', 'G_occipital_middle', 'S_temporal_sup', 'S_front_inf',
        #                 'G_front_inf-Opercular', 'S_precentral-inf-part', 'G_temporal_inf',
        #                 'G_pariet_inf-Supramar', 'G_temp_sup-Plan_tempo', 'S_interm_prim-Jensen', 'G_temp_sup-Lateral'], # , 'G_front_inf-Orbital',  'G_orbital',
        #     "ventral": ['S_oc-temp_lat', 'G_temporal_inf', 'G_oc-temp_lat-fusifor',
        #                 'Pole_temporal'], #, 'G_front_inf-Orbital', 'G_orbital',
        #     "posterior": ['G_pariet_inf-Angular', 'S_temporal_sup', 'G_parietal_sup'] #, 'S_intrapariet_and_P_trans' , 'G_occipital_sup'
        # }
        rois_for_view = {
            METRIC_MOD_AGNOSTIC_AND_CROSS: {
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
            ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC: {
                "left": {
                    "medial": ['precuneus', 'isthmuscingulate', 'parahippocampal'],
                    "lateral": ['inferiorparietal', 'supramarginal', 'middletemporal', 'bankssts'],
                    "ventral": ['inferiortemporal', 'fusiform'],
                },
                "right": {
                    "medial": ['precuneus', 'isthmuscingulate', 'parahippocampal'],
                    "lateral": ['inferiorparietal', 'middletemporal', 'bankssts'],
                    "ventral": ['inferiortemporal', 'fusiform'],
                },
            },
            ACC_IMAGERY_MOD_AGNOSTIC: {
                "left": {
                    "medial": ['precuneus', 'isthmuscingulate', 'parahippocampal'],
                    "lateral": ['inferiorparietal', 'supramarginal', 'middletemporal', 'bankssts'],
                    "ventral": ['inferiortemporal', 'fusiform'],
                },
                "right": {
                    "medial": ['precuneus', 'isthmuscingulate', 'parahippocampal'],
                    "lateral": ['inferiorparietal', 'middletemporal', 'bankssts'],
                    "ventral": ['inferiortemporal', 'fusiform'],
                },
            }
        }

        result_values = dict()

        if result_metric == METRIC_MOD_AGNOSTIC_AND_CROSS:
            tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
            orig_result_values = pickle.load(open(tfce_values_path, "rb"))
            for hemi in HEMIS:
                result_values[hemi] = orig_result_values[hemi][args.metric]

            null_distribution_tfce_values_file = os.path.join(
                permutation_results_dir(args),
                f"tfce_values_null_distribution{get_hparam_suffix(args)}.p"
            )
            null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))
            significance_cutoff, _ = calc_significance_cutoff(null_distribution_tfce_values, args.metric,
                                                              args.p_value_threshold)
            # significance_cutoff = 2333.16
            print(f"{result_metric} significance cutoff: {significance_cutoff}")
            threshold = significance_cutoff
            cbar_min = 0
            # cbar_max = CBAR_TFCE_MAX_VALUE
            cbar_max = np.nanmax(np.concatenate((result_values['left'], result_values['right'])))
            # print(f"{result_metric} max tfce value across hemis: {cbar_max}")

        elif result_metric.startswith("pairwise_acc"):
            # cbar_min = ACC_COLORBAR_MIN
            # cbar_max = COLORBAR_MAX
            # threshold = ACC_COLORBAR_THRESHOLD
            # subject_scores = load_per_subject_scores(args)
            # for hemi in HEMIS:
            #     score_hemi_avgd = np.nanmean([subject_scores[subj][hemi][result_metric] for subj in args.subjects],
            #                                  axis=0)
            #     result_values[hemi] = score_hemi_avgd

            # t_values = pickle.load(open(os.path.join(permutation_results_dir(args), "t_values.p"), 'rb'))
            # for hemi in HEMIS:
            #     result_values[hemi] = t_values[hemi][args.metric]
            tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
            orig_result_values = pickle.load(open(tfce_values_path, "rb"))
            for hemi in HEMIS:
                result_values[hemi] = orig_result_values[hemi][args.metric]

            null_distribution_tfce_values_file = os.path.join(
                permutation_results_dir(args),
                f"tfce_values_null_distribution{get_hparam_suffix(args)}.p"
            )
            null_distribution_tfce_values = pickle.load(open(null_distribution_tfce_values_file, 'rb'))
            significance_cutoff, _ = calc_significance_cutoff(null_distribution_tfce_values, args.metric,
                                                              args.p_value_threshold)
            threshold = significance_cutoff
            cbar_min = 0
            cbar_max = np.nanmax(np.concatenate((result_values['left'], result_values['right'])))
            # print(f"{result_metric} max tfce value across hemis: {np.nanmax(np.concatenate((result_values['left'], result_values['right'])))}")

            # for hemi in HEMIS:
            #     print(f"{hemi} hemi fraction of values above thresh: {np.mean(result_values[hemi] > significance_cutoffs[hemi])}")
            #     result_values[hemi][result_values[hemi] < significance_cutoffs[hemi]] = 0
            #     print([round(val) for val in result_values[hemi][result_values[hemi] > 2000][:20]])
            # threshold = 1

            # from t-val table:
            # for p<0.05: 2.015
            # for p<0.01: 3.365
            # for p<0.001: 5.893

            # from permutation testing:
            # test statistic significance cutoff for p<0.05: 2.06
            # min mean acc:
            # 0.5308641975308642
            # test statistic significance cutoff for p<0.01: 3.44
            # min mean acc:
            # 0.5740740740740741
            # test statistic significance cutoff for p<0.001: 6.03
            # min mean acc: 0.5902777777777778
            # threshold = 2.015
            # cbar_min = 0
            # cbar_max = CBAR_T_VAL_MAX

        else:
            raise RuntimeError(f"Unknown metric: {result_metric}")

        print(f"{result_metric} cbar max: {cbar_max}")

        for hemi in HEMIS:
            hemi_fs = FS_HEMI_NAMES[hemi]
            # atlas_path = os.path.join(FREESURFER_HOME_DIR, f"subjects/fsaverage/label/{hemi_fs}.aparc.a2009s.annot")
            atlas_path = os.path.join(FREESURFER_HOME_DIR, f"subjects/fsaverage/label/{hemi_fs}.aparc.annot")
            atlas_labels, atlas_colors, names = nibabel.freesurfer.read_annot(atlas_path)
            names = [name.decode() for name in names]

            # subcortical_atlas_path = os.path.join(ROOT_DIR, f"atlas_data/{hemi}_subcortical.annot")
            # subcortical_atlas_path = os.path.join(ROOT_DIR, f"atlas_data/{hemi_fs}.test.annot")
            # subcortical_atlas_labels, subcortical_atlas_colors, subcortical_names = nibabel.freesurfer.read_annot(subcortical_atlas_path)
            # subcortical_names = [name.decode() for name in subcortical_names]
            # names = names + subcortical_names
            # atlas_colors = np.vstack((atlas_colors, subcortical_atlas_colors))
            # add labels from subcortical atlas
            # offset = np.max(atlas_labels) + 1
            # label_to_id = {id: id if id == -1 else i + offset for i, id in enumerate(label_ids)}
            # subcortical_atlas_labels_transformed = np.array([-1 if l == -1 else l + offset for l in subcortical_atlas_labels])
            # subcortical_atlas_labels_transformed = np.array([l + offset for l in subcortical_atlas_labels])
            # atlas_labels[atlas_labels == -1] = subcortical_atlas_labels_transformed[atlas_labels == -1]

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
    for result_metric in METRICS:
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
    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    plot(args)
    create_composite_image(args)
