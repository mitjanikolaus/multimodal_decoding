import argparse

import matplotlib
import nibabel.freesurfer
import numpy as np
from PIL import Image
from matplotlib.colors import Normalize, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle

import seaborn as sns
from nilearn._utils import compare_version
from nilearn.plotting import plot_surf
from nilearn.plotting.surf_plotting import _get_faces_on_edge
from nilearn.surface import load_surf_mesh
from nilearn.surface.surface import check_extensions, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS, load_surf_data

from analyses.searchlight.searchlight import METRIC_IMAGES, METRIC_CAPTIONS, METRIC_DIFF_IMAGES, METRIC_DIFF_CAPTIONS
from analyses.searchlight.searchlight_permutation_testing import METRIC_MIN, permutation_results_dir, \
    get_hparam_suffix
from analyses.searchlight.searchlight_results_plotting import CMAP_POS_ONLY, DEFAULT_VIEWS, save_plot_and_crop_img, \
    P_VALUE_THRESHOLD, append_images
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


def save_legend(legend_dict, p_values_atlas_results_dir):
    patches = [mpatches.Patch(color=color, label=label) for label, color in legend_dict.items()]

    plt.figure(figsize=(30, 10))
    plt.legend(handles=patches, ncol=2, prop={'size': 12})
    plt.gca().set_axis_off()

    path = os.path.join(p_values_atlas_results_dir, f"legend.png")

    plt.savefig(path, dpi=300, transparent=True)
    image = Image.open(path)
    image = image.crop(image.getbbox())
    image.save(path)
    plt.close()


def _get_faces_within_edge(faces, parc_idx):
    """Identify which faces lie within a parcellation ROI \
    defined by the indices in parc_idx.

    Parameters
    ----------
    faces : numpy.ndarray of shape (n, 3), indices of the mesh faces

    parc_idx : numpy.ndarray, indices of the vertices
        of the region to be plotted

    """
    # count how many vertices belong to the given parcellation in each face
    verts_per_face = np.isin(faces, parc_idx).sum(axis=1)

    return verts_per_face >= 1


def plot_surf_contours_filled(surf_mesh, roi_map, axes=None, figure=None, levels=None,
                              labels=None, colors=None, legend=False, cmap='tab20',
                              title=None, output_file=None, foreground_alpha=0.5, **kwargs):
    """Plot ROIs on a surface, optionally over a statistical map.
    """
    if figure is None and axes is None:
        figure = plot_surf(surf_mesh, **kwargs)
        axes = figure.axes[0]
    if figure is None:
        figure = axes.get_figure()
    if axes is None:
        axes = figure.axes[0]
    if axes.name != '3d':
        raise ValueError('Axes must be 3D.')
    # test if axes contains Poly3DCollection, if not initialize surface
    if not axes.collections or not isinstance(axes.collections[0],
                                              Poly3DCollection):
        _ = plot_surf(surf_mesh, axes=axes, **kwargs)

    _, faces = load_surf_mesh(surf_mesh)

    check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)
    roi = load_surf_data(roi_map)

    if levels is None:
        levels = np.unique(roi_map)
    if colors is None:
        n_levels = len(levels)
        vmax = n_levels
        cmap = plt.get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=vmax)
        colors = [cmap(norm(color_i)) for color_i in range(vmax)]
    else:
        try:
            colors = [to_rgba(color, alpha=foreground_alpha) for color in colors]
        except ValueError:
            raise ValueError('All elements of colors need to be either a'
                             ' matplotlib color string or RGBA values.')

    if labels is None:
        labels = [None] * len(levels)
    if not (len(levels) == len(labels) == len(colors)):
        raise ValueError('Levels, labels, and colors '
                         'argument need to be either the same length or None.')

    patch_list = []
    for level, color, label in zip(levels, colors, labels):
        roi_indices = np.where(roi == level)[0]
        faces_outside = _get_faces_within_edge(faces, roi_indices)
        # Fix: Matplotlib version 3.3.2 to 3.3.3
        # Attribute _facecolors3d changed to _facecolor3d in
        # matplotlib version 3.3.3
        if compare_version(matplotlib.__version__, "<", "3.3.3"):
            axes.collections[0]._facecolors3d[faces_outside] = color
            if axes.collections[0]._edgecolors3d.size == 0:
                axes.collections[0].set_edgecolor(
                    axes.collections[0]._facecolors3d
                )
            axes.collections[0]._edgecolors3d[faces_outside] = color
        else:
            axes.collections[0]._facecolor3d[faces_outside] = color
            if axes.collections[0]._edgecolor3d.size == 0:
                axes.collections[0].set_edgecolor(
                    axes.collections[0]._facecolor3d
                )
            axes.collections[0]._edgecolor3d[faces_outside] = color
        if label and legend:
            patch_list.append(mpatches.Patch(color=color, label=label))
    # plot legend only if indicated and labels provided
    if legend and np.any([lbl is not None for lbl in labels]):
        figure.legend(handles=patch_list)
        # if legends, then move title to the left
    if title is None and hasattr(figure._suptitle, "_text"):
        title = figure._suptitle._text
    if title:
        axes.set_title(title)
    # save figure if output file is given
    if output_file is not None:
        figure.savefig(output_file)
        plt.close(figure)
    else:
        return figure


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
        "medial": ['G_precuneus', 'S_subparietal', 'G_cingul-Post-dorsal', 'S_parieto_occipital'],
        # , 'Left-Hippocampus'
        "lateral": ['G_pariet_inf-Angular', 'G_occipital_middle', 'G_temporal_inf', 'S_temporal_sup'],
        "ventral": ['S_oc-temp_lat', 'G_oc-temp_lat-fusifor']  # , 'G_temporal_inf']
    }

    unique_rois = set()
    for r in rois_for_view.values():
        unique_rois.update(r)

    label_names_dict = destrieux_label_names()
    all_colors = {label_names_dict[r]: c for r, c in
                  zip(unique_rois, sns.color_palette(n_colors=len(unique_rois) + 1)[1:])}

    save_legend(all_colors, p_values_atlas_results_dir)
    # plt.savefig(path)

    for hemi in HEMIS:
        hemi_fs = FS_HEMI_NAMES[hemi]
        resolution_fs = "fsaverage" if args.resolution == "fsaverage7" else args.resolution
        atlas_path = os.path.join(FREESURFER_HOME_DIR, f"subjects/{resolution_fs}/label/{hemi_fs}.aparc.a2009s.annot")
        atlas_labels, atlas_colors, names = nibabel.freesurfer.read_annot(atlas_path)
        names = [name.decode() for name in names]

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
            colors = [all_colors[l] for l in label_names]

            scores_hemi = p_values[hemi]
            fig = plotting.plot_surf_stat_map(
                fsaverage[f"infl_{hemi}"],
                scores_hemi,
                bg_map=fsaverage[f"sulc_{hemi}"],
                bg_on_data=True,
                hemi=hemi,
                view=view,
                colorbar=False,
                threshold=-np.log10(P_VALUE_THRESHOLD),
                vmax=cbar_max,
                vmin=cbar_min,
                cmap=CMAP_POS_ONLY,
                symmetric_cbar=False,
                darkness=0.7,
            )

            plot_surf_contours_filled(
                fsaverage[f"infl_{hemi}"], atlas_labels, labels=label_names,
                levels=regions_indices, figure=fig,
                legend=False, colors=colors, foreground_alpha=0.05
            )

            title = f"{view}_{hemi}"
            path = os.path.join(p_values_atlas_results_dir, f"{title}.png")
            save_plot_and_crop_img(path)
            print(f"saved {path}")


def create_composite_image(args):
    results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution))
    p_values_imgs_dir = str(os.path.join(results_path, "tmp", "p_values_atlas"))

    images_lateral = [Image.open(os.path.join(p_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["lateral"] for hemi in HEMIS]
    images_medial = [Image.open(os.path.join(p_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["medial"] for hemi in HEMIS]

    imgs_ventral = [Image.open(os.path.join(p_values_imgs_dir, f"ventral_{hemi}.png")) for hemi in HEMIS]
    img_ventral = append_images(images=imgs_ventral, horizontally=False)

    img_medial = append_images(images=images_medial)

    img_row_2 = append_images(images=[img_medial, img_ventral])

    img_colorbar = Image.open(os.path.join(p_values_imgs_dir, "colorbar.png"))
    img_lateral = append_images(images=images_lateral)

    img_row_1 = append_images([img_lateral, img_colorbar], padding=20)

    roi_legend = Image.open(os.path.join(p_values_imgs_dir, f"legend.png"))

    p_val_image = append_images([img_row_1, img_row_2, roi_legend], padding=5, horizontally=False)

    path = os.path.join(results_path, "searchlight_results.png")
    p_val_image.save(path, transparent=True)
    print("done")


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
    create_composite_image(args)
