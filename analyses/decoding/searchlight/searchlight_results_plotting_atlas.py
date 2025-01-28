import argparse

import nibabel.freesurfer
import numpy as np
from PIL import Image
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import os
import pickle

from analyses.cluster_analysis import calc_significance_cutoff
from analyses.decoding.searchlight.searchlight_permutation_testing import METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, permutation_results_dir, \
    get_hparam_suffix
from analyses.decoding.searchlight.searchlight_results_plotting import DEFAULT_VIEWS, save_plot_and_crop_img, \
    append_images
from analyses.visualization.plotting_utils import plot_surf_contours_custom, plot_surf_stat_map_custom
from eval import ACC_IMAGERY
from utils import RESULTS_DIR, HEMIS, FREESURFER_HOME_DIR, FS_HEMI_NAMES, DEFAULT_RESOLUTION, SUBJECTS, \
    METRIC_CROSS_DECODING, DEFAULT_MODEL

HCP_ATLAS_DIR = os.path.join("atlas_data", "hcp_surface")
HCP_ATLAS_LH = os.path.join(HCP_ATLAS_DIR, "lh.HCP-MMP1.annot")
HCP_ATLAS_RH = os.path.join(HCP_ATLAS_DIR, "rh.HCP-MMP1.annot")

CMAP_POS_ONLY = "hot"



def plot(args):
    plt.style.use("dark_background")

    for result_metric in [METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, ACC_IMAGERY, METRIC_CROSS_DECODING]:
        results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution, args.mode))
        atlas_tmp_results_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))
        os.makedirs(atlas_tmp_results_dir, exist_ok=True)

        args.metric = result_metric
        significance_cutoff, _ = calc_significance_cutoff(args, args.p_value_threshold) # (603, None)

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
            "left": {
                "medial": ['precuneus', 'isthmuscingulate'],
                "lateral": ['inferiorparietal', 'supramarginal', 'middletemporal', 'parsopercularis',
                            'rostralmiddlefrontal', 'precentral'],
                "ventral": ['inferiortemporal', 'fusiform'],
            },
            "right": {
                "medial": ['precuneus', 'isthmuscingulate'],
                "lateral": ['inferiorparietal', 'rostralmiddlefrontal'],
                "ventral": ['inferiortemporal'],
            }
        }

        # unique_rois = set()
        # for r in rois_for_view.values():
        #     unique_rois.update(r)

        # label_names_dict = destrieux_label_names()
        # color_palette = [(183, 242, 34), (127, 176, 4),
        #                  (174, 245, 176), (10, 250, 16), (4, 186, 8), (2, 110, 5), (1, 74, 3),
        #                  (193, 247, 233), (5, 245, 183), (1, 140, 104),
        #                  (145, 231, 242), (5, 220, 247), (0, 120, 135),
        #                  (115, 137, 245), (7, 48, 245), (2, 29, 158),
        #                  (174, 92, 237), (140, 7, 242), (76, 3, 133),
        #                  (245, 105, 242), (250, 5, 245), (125, 2, 122),
        #                  (242, 34, 152)]
        # color_palette = [(x[0] / 255, x[1] / 255, x[2] / 255) for x in color_palette]
        #
        # assert len(unique_rois) <= len(color_palette), f"not enough colors for {len(unique_rois)} ROIS"
        #
        # all_colors = {label_names_dict[r]: c for r, c in
        #               zip(unique_rois, color_palette)}
        #
        # save_legend(all_colors, tfce_values_atlas_results_dir)

        tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values{get_hparam_suffix(args)}.p")
        result_values = pickle.load(open(tfce_values_path, "rb"))
        for hemi in HEMIS:
            result_values[hemi] = result_values[hemi][args.metric]
        # elif result_metric == 'imagery':
        #     result_values = {}
        #     subject_scores = load_per_subject_scores(args)
        #     for hemi in HEMIS:
        #         score_hemi_avgd = np.nanmean([subject_scores[subj][hemi][ACC_IMAGERY_WHOLE_TEST] for subj in args.subjects], axis=0)
        #         result_values[hemi] = score_hemi_avgd
        # else:
        #     raise RuntimeError(f"Unknown metric: {result_metric}")

        cbar_max = np.nanmax(np.concatenate((result_values['left'], result_values['right'])))
        cbar_min = 0
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



            for i, (view, rois) in enumerate(rois_for_view[hemi].items()):
                regions_indices = [names.index(roi) for roi in rois]
                # label_names = [label_names_dict[r] if r in label_names_dict else r.replace("-", " ") for r in rois]
                label_names = [r for r in rois]
                # colors = [all_colors[label_names[regions_indices.index(i)]] if i in regions_indices else (0, 0, 0, 0) for i
                #           in range(np.nanmax(atlas_labels))]
                # cmap = ListedColormap(colors)
                atlas_labels_current_view = np.array([l if l in regions_indices else np.nan for l in atlas_labels])
                # fig = plot_surf_roi_custom(
                #     fsaverage[f"infl_{hemi}"], roi_map=atlas_labels_current_view,
                #     bg_map=fsaverage[f"sulc_{hemi}"], hemi=hemi,
                #     view=view, alpha=1, cmap=cmap,
                #     bg_on_data=True, darkness=0.4, categorical_cmap=True,
                # )
                # plot_surf_stat_map_custom(
                #     fsaverage[f"infl_{hemi}"],
                #     tfce_values[hemi][args.metric],
                #     hemi=hemi,
                #     view=view,
                #     colorbar=False,
                #     threshold=significance_cutoff,
                #     vmax=cbar_max,
                #     vmin=cbar_min,
                #     cmap=CMAP_POS_ONLY,
                #     figure=fig,
                #     keep_bg=True,
                # )

                fig = plotting.plot_surf_stat_map(
                    fsaverage[f"infl_{hemi}"],
                    result_values[hemi],
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    hemi=hemi,
                    view=view,
                    colorbar=False,
                    threshold=significance_cutoff,
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
                    colors=['w'] * len(regions_indices),
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
            threshold=significance_cutoff,
            vmax=cbar_max,
            vmin=cbar_min,
            cmap=CMAP_POS_ONLY,
            figure=fig,
            metric=result_metric,
        )
        save_plot_and_crop_img(os.path.join(atlas_tmp_results_dir, "colorbar.png"), crop_cbar=True, horizontal_cbar=True)


def create_composite_image(args):
    for result_metric in [METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, ACC_IMAGERY, METRIC_CROSS_DECODING]:
        results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution, args.mode))
        results_values_imgs_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))

        images_lateral = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["lateral"] for hemi
                          in HEMIS]
        images_medial = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["medial"] for hemi
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
        p_val_image.save(path, transparent=True, facecolor="black")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--features", type=str, default="avg_test_avg")

    parser.add_argument("--mod-specific-vision-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-vision-features", type=str, default="vision_test_vision")

    parser.add_argument("--mod-specific-lang-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-lang-features", type=str, default="lang_test_lang")

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--mode", type=str, default='n_neighbors_200')

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--metric", type=str, default=METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    plot(args)
    create_composite_image(args)
