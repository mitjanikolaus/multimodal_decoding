import argparse

import nibabel.freesurfer
import numpy as np
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle

import seaborn as sns

from analyses.searchlight.searchlight_permutation_testing import METRIC_MIN, permutation_results_dir, \
    get_hparam_suffix
from analyses.searchlight.searchlight_results_plotting import CMAP_POS_ONLY, DEFAULT_VIEWS, save_plot_and_crop_img, \
    P_VALUE_THRESHOLD
from utils import RESULTS_DIR, HEMIS, FREESURFER_HOME_DIR, HEMIS_FS, FS_HEMI_NAMES, ROOT_DIR

HCP_ATLAS_DIR = os.path.join("atlas_data", "hcp_surface")
HCP_ATLAS_LH = os.path.join(HCP_ATLAS_DIR, "lh.HCP-MMP1.annot")
HCP_ATLAS_RH = os.path.join(HCP_ATLAS_DIR, "rh.HCP-MMP1.annot")


def destrieux_label_names():
    long_names = {}
    with open("atlas_data/destrieux.txt") as file:
        for line in file:
            line = line.rstrip()
            long_name = " ".join(", ".join(line.split(', ')[1:]).split(' ')[1:])
            name = line.split(', ')[1].split(' ')[0]
            long_names[name] = long_name

    return long_names


def plot(args):
    results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution))
    p_values_atlas_results_dir = str(os.path.join(results_path, "tmp", "p_values_atlas"))
    os.makedirs(p_values_atlas_results_dir, exist_ok=True)

    p_values_path = os.path.join(permutation_results_dir(args), f"p_values{get_hparam_suffix(args)}.p")
    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    # labels, colors, names = nibabel.freesurfer.read_annot(HCP_ATLAS_LH)
    # destrieux_atlas = datasets.fetch_atlas_surf_destrieux()

    rois_for_view = {
        "medial": ['G_precuneus', 'S_subparietal', 'G_cingul-Post-dorsal', 'S_parieto_occipital'], #, 'Left-Hippocampus'
        "lateral": ['G_pariet_inf-Angular', 'G_occipital_middle', 'G_temporal_inf', 'S_temporal_sup'],
        "ventral": ['S_oc-temp_lat', 'G_oc-temp_lat-fusifor'] #, 'G_temporal_inf']
    }

    unique_rois = set()
    for r in rois_for_view.values():
        unique_rois.update(r)
    all_colors = {r: c for r, c in zip(unique_rois, sns.color_palette(n_colors=len(unique_rois) + 1)[1:])}

    patches = [mpatches.Patch(color=color, label=label) for label, color in all_colors.items()]
    plt.legend(handles=patches)
    path = os.path.join(p_values_atlas_results_dir, f"legend.png")
    plt.savefig(path)

    for hemi in HEMIS:
        hemi_fs = FS_HEMI_NAMES[hemi]
        resolution_fs = "fsaverage" if args.resolution == "fsaverage7" else args.resolution
        atlas_path = os.path.join(FREESURFER_HOME_DIR, f"subjects/{resolution_fs}/label/{hemi_fs}.aparc.a2009s.annot")
        atlas_labels, atlas_colors, names = nibabel.freesurfer.read_annot(atlas_path)
        names = [name.decode() for name in names]
        label_names_dict = destrieux_label_names()

        # subcortical_atlas_path = os.path.join(ROOT_DIR, f"atlas_data/{hemi}_subcortical.annot")
        subcortical_atlas_path = os.path.join(ROOT_DIR, f"atlas_data/{hemi_fs}.test.annot")

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

        cbar_max = np.nanmax(np.concatenate((p_values['left'], p_values['right'])))
        cbar_min = 0
        for i, (view, rois) in enumerate(rois_for_view.items()):
            regions_indices = [names.index(roi) for roi in rois]
            label_names = [label_names_dict[r] if r in label_names_dict else r.replace("-", " ") for r in rois]
            # colors = [atlas_colors[i][:4] / 255 for i in regions_indices]
            # colors = [(r, g, b, 0.5) for (r, g, b, a) in colors]
            colors = [all_colors[r] for r in rois]
            print(colors)

            scores_hemi = p_values[hemi]
            fig_hemi = plotting.plot_surf_stat_map(
                fsaverage[f"infl_{hemi}"],
                scores_hemi,
                hemi=hemi,
                view=view,
                bg_map=fsaverage[f"sulc_{hemi}"],
                bg_on_data=True,
                colorbar=False,
                threshold=-np.log10(P_VALUE_THRESHOLD),
                vmax=cbar_max,
                vmin=cbar_min,
                cmap=CMAP_POS_ONLY,
                symmetric_cbar=False,
            )

            plotting.plot_surf_contours(fsaverage[f"infl_{hemi}"], atlas_labels, labels=label_names,
                                        levels=regions_indices, figure=fig_hemi,
                                        legend=False,
                                        colors=colors)
            title = f"{view}_{hemi}"
            path = os.path.join(p_values_atlas_results_dir, f"{title}.png")
            save_plot_and_crop_img(path)
            print(f"saved {path}")

# def save_separate_img_and_legend()
# save_plot_and_crop_img

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='imagebind')
    parser.add_argument("--features", type=str, default="avg_test_avg")

    parser.add_argument("--mod-specific-vision-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-vision-features", type=str, default="vision_test_vision")

    parser.add_argument("--mod-specific-lang-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-lang-features", type=str, default="lang_test_lang")

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default='fsaverage7')
    parser.add_argument("--mode", type=str, default='n_neighbors_200')

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--metric", type=str, default=METRIC_MIN)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    plot(args)
