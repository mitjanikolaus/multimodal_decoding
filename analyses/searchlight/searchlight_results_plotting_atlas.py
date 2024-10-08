import argparse
from warnings import warn

import nibabel.freesurfer
import numpy as np
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import make_axes
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle

import seaborn as sns
from nilearn.plotting.cm import mix_colormaps
from nilearn.plotting.img_plotting import get_colorbar_and_data_ranges
from nilearn.plotting.surf_plotting import _get_cmap_matplotlib, \
    _get_ticks_matplotlib, _threshold_and_rescale, _compute_surf_map_faces_matplotlib, _get_view_plot_surf_matplotlib, \
    _compute_facecolors_matplotlib
from nilearn.surface import load_surf_mesh
from nilearn.surface.surface import check_extensions, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS, load_surf_data

from analyses.searchlight.searchlight_permutation_testing import METRIC_MIN, permutation_results_dir, \
    get_hparam_suffix
from analyses.searchlight.searchlight_results_plotting import CMAP_POS_ONLY, DEFAULT_VIEWS, save_plot_and_crop_img, \
    P_VALUE_THRESHOLD, append_images
from preprocessing.transform_to_surface import DEFAULT_RESOLUTION
from utils import RESULTS_DIR, HEMIS, FREESURFER_HOME_DIR, FS_HEMI_NAMES, ROOT_DIR

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


def _plot_surf_matplotlib_custom(coords, faces, surf_map=None, bg_map=None, bg_on_data=False, keep_bg=False,
                                 hemi='left', view='lateral', cmap=None,
                                 colorbar=False, avg_method='mean', threshold=None,
                                 alpha='auto', vmin=None, vmax=None, cbar_vmin=None,
                                 cbar_vmax=None, cbar_tick_format='%.2g',
                                 title=None, output_file=None, darkness=0.7,
                                 axes=None, figure=None, categorical_cmap=False):
    """Help for plot_surf.

    This function handles surface plotting when the selected
    engine is matplotlib.
    """
    _default_figsize = [4, 5]
    limits = [coords.min(), coords.max()]

    # Get elevation and azimut from view
    elev, azim = _get_view_plot_surf_matplotlib(hemi, view)

    # if no cmap is given, set to matplotlib default
    if cmap is None:
        cmap = plt.get_cmap(plt.rcParamsDefault['image.cmap'])
    # if cmap is given as string, translate to matplotlib cmap
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    figsize = _default_figsize
    # Leave space for colorbar
    if colorbar:
        figsize[0] += .7
    # initiate figure and 3d axes
    if axes is None:
        if figure is None:
            figure = plt.figure(figsize=figsize)
        axes = figure.add_axes((0, 0, 1, 1), projection="3d")
    else:
        if figure is None:
            figure = axes.get_figure()
    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()

    # plot mesh without data
    p3dcollec = axes.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                  triangles=faces, linewidth=0.1,
                                  antialiased=False,
                                  color='white')

    # reduce viewing distance to remove space around mesh
    axes.set_box_aspect(None, zoom=1.3)

    if keep_bg:
        bg_face_colors = figure.axes[0].collections[0]._facecolor3d
    else:
        bg_face_colors = _compute_facecolors_matplotlib(
            bg_map, faces, coords.shape[0], darkness, alpha
        )
    if surf_map is not None:
        surf_map_faces = _compute_surf_map_faces_matplotlib(
            surf_map, faces, avg_method, coords.shape[0],
            bg_face_colors.shape[0]
        )

        if categorical_cmap:
            surf_map_face_colors = cmap(surf_map_faces.astype(int))
            surf_map_faces, kept_indices, vmin, vmax = _threshold_and_rescale(
                surf_map_faces, threshold, vmin, vmax
            )
        else:
            surf_map_faces, kept_indices, vmin, vmax = _threshold_and_rescale(
                surf_map_faces, threshold, vmin, vmax
            )
            surf_map_face_colors = cmap(surf_map_faces)

        # set transparency of voxels under threshold to 0
        surf_map_face_colors[~kept_indices, 3] = 0

        if bg_on_data:
            # if need be, set transparency of voxels above threshold to 0.7
            # so that background map becomes visible
            surf_map_face_colors[kept_indices, 3] = 0.7

        face_colors = mix_colormaps(
            surf_map_face_colors,
            bg_face_colors
        )

        if colorbar:
            cbar_vmin = cbar_vmin if cbar_vmin is not None else vmin
            cbar_vmax = cbar_vmax if cbar_vmax is not None else vmax
            ticks = _get_ticks_matplotlib(cbar_vmin, cbar_vmax,
                                          cbar_tick_format, threshold)
            our_cmap, norm = _get_cmap_matplotlib(cmap,
                                                  vmin,
                                                  vmax,
                                                  cbar_tick_format,
                                                  threshold)
            bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)

            # we need to create a proxy mappable
            proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(surf_map_faces)
            cax, _ = make_axes(axes, location='right', fraction=.15,
                               shrink=.5, pad=.0, aspect=10.)
            figure.colorbar(
                proxy_mappable, cax=cax, ticks=ticks,
                boundaries=bounds, spacing='proportional',
                format=cbar_tick_format, orientation='vertical')

        p3dcollec.set_facecolors(face_colors)
        p3dcollec.set_edgecolors(face_colors)

    if title is not None:
        axes.set_title(title)
    # save figure if output file is given
    if output_file is not None:
        figure.savefig(output_file)
        plt.close()
    else:
        return figure


def plot_surf_custom(
        surf_mesh, surf_map=None, hemi='left', view='lateral', engine='matplotlib', cmap=None, symmetric_cmap=False,
        colorbar=False, avg_method=None, threshold=None, alpha=None, vmin=None, vmax=None, cbar_vmin=None,
        cbar_vmax=None, cbar_tick_format="auto", title=None, title_font_size=18, output_file=None, axes=None,
        figure=None, bg_map=None, bg_on_data=False, keep_bg=False, darkness=0.7, categorical_cmap=False,
):
    """Plot surfaces with optional background and data."""

    parameters_not_implemented_in_plotly = {
        "avg_method": avg_method,
        "figure": figure,
        "axes": axes,
        "cbar_vmin": cbar_vmin,
        "cbar_vmax": cbar_vmax,
        "alpha": alpha
    }

    check_extensions(surf_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

    if engine == 'plotly':
        for parameter, value in parameters_not_implemented_in_plotly.items():
            if value is not None:
                warn(
                    (f"'{parameter}' is not implemented "
                     "for the plotly engine.\n"
                     f"Got '{parameter} = {value}'.\n"
                     f"Use '{parameter} = None' to silence this warning."))

    coords, faces = load_surf_mesh(surf_mesh)

    if engine == 'matplotlib':
        # setting defaults
        if avg_method is None:
            avg_method = 'mean'
        if alpha is None:
            alpha = 'auto'

        if cbar_tick_format == "auto":
            cbar_tick_format = '%.2g'
        fig = _plot_surf_matplotlib_custom(
            coords, faces, surf_map=surf_map, hemi=hemi,
            view=view, cmap=cmap, colorbar=colorbar, avg_method=avg_method,
            threshold=threshold, alpha=alpha, darkness=darkness,
            vmin=vmin, vmax=vmax, cbar_vmin=cbar_vmin,
            cbar_vmax=cbar_vmax, cbar_tick_format=cbar_tick_format,
            title=title, bg_map=bg_map, bg_on_data=bg_on_data, keep_bg=keep_bg,
            output_file=output_file, axes=axes, figure=figure, categorical_cmap=categorical_cmap)

    elif engine == 'plotly':
        raise NotImplementedError()

    else:
        raise ValueError(f"Unknown plotting engine {engine}. "
                         "Please use either 'matplotlib' or "
                         "'plotly'.")

    return fig


def plot_surf_stat_map_custom(
        surf_mesh, stat_map, hemi='left', view='lateral', engine='matplotlib', threshold=None, alpha=None, vmin=None,
        vmax=None, cmap='cold_hot', colorbar=True, symmetric_cbar="auto", cbar_tick_format="auto", title=None,
        title_font_size=18, output_file=None, axes=None, figure=None, avg_method=None,
        bg_map=None, bg_on_data=False, keep_bg=False, categorical_cmap=False, **kwargs
):
    """Plot a stats map on a surface :term:`mesh` with optional background.    """
    check_extensions(stat_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)
    loaded_stat_map = load_surf_data(stat_map)

    # Call get_colorbar_and_data_ranges to derive symmetric vmin, vmax
    # And colorbar limits depending on symmetric_cbar settings
    cbar_vmin, cbar_vmax, vmin, vmax = get_colorbar_and_data_ranges(
        loaded_stat_map,
        vmin=vmin,
        vmax=vmax,
        symmetric_cbar=symmetric_cbar,
    )
    # Set to None the values that are not used by plotly
    # to avoid warnings thrown by plot_surf
    if engine == "plotly":
        cbar_vmin = None
        cbar_vmax = None

    display = plot_surf_custom(
        surf_mesh, surf_map=loaded_stat_map,
        hemi=hemi,
        view=view, engine=engine, avg_method=avg_method, threshold=threshold,
        cmap=cmap, symmetric_cmap=True, colorbar=colorbar,
        cbar_tick_format=cbar_tick_format, alpha=alpha,
        vmax=vmax, vmin=vmin,
        title=title, title_font_size=title_font_size, output_file=output_file,
        axes=axes, figure=figure, cbar_vmin=cbar_vmin,
        bg_map=bg_map, bg_on_data=bg_on_data, keep_bg=keep_bg,
        cbar_vmax=cbar_vmax, categorical_cmap=categorical_cmap, **kwargs
    )
    return display


def plot_surf_roi_custom(surf_mesh,
                         roi_map,
                         bg_map=None,
                         hemi='left',
                         view='lateral',
                         engine='matplotlib',
                         avg_method=None,
                         threshold=1e-14,
                         alpha=None,
                         vmin=None,
                         vmax=None,
                         cmap=None,
                         cbar_tick_format="auto",
                         bg_on_data=False,
                         title=None,
                         title_font_size=18,
                         output_file=None,
                         axes=None,
                         figure=None,
                         darkness=0.7,
                         categorical_cmap=False,
                         **kwargs):
    """Plot ROI on a surface :term:`mesh` with optional background."""

    if engine == "matplotlib" and avg_method is None:
        avg_method = "median"

    # preload roi and mesh to determine vmin, vmax and give more useful error
    # messages in case of wrong inputs
    check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)
    roi = load_surf_data(roi_map)

    idx_not_na = ~np.isnan(roi)
    if vmin is None:
        vmin = np.nanmin(roi)
    if vmax is None:
        vmax = 1 + np.nanmax(roi)

    mesh = load_surf_mesh(surf_mesh)

    if roi.ndim != 1:
        raise ValueError('roi_map can only have one dimension but has '
                         f'{roi.ndim} dimensions')
    if roi.shape[0] != mesh[0].shape[0]:
        raise ValueError('roi_map does not have the same number of vertices '
                         'as the mesh. If you have a list of indices for the '
                         'ROI you can convert them into a ROI map like this:\n'
                         'roi_map = np.zeros(n_vertices)\n'
                         'roi_map[roi_idx] = 1')
    if (roi < 0).any():
        warn(
            (
                'Negative values in roi_map will no longer be allowed in'
                ' Nilearn version 0.13'
            ),
            DeprecationWarning,
        )
    if not np.array_equal(roi[idx_not_na], roi[idx_not_na].astype(int)):
        warn(
            (
                'Non-integer values in roi_map will no longer be allowed in'
                ' Nilearn version 0.13'
            ),
            DeprecationWarning,
        )

    if cbar_tick_format == "auto":
        cbar_tick_format = "." if engine == "plotly" else "%i"

    display = plot_surf_custom(mesh,
                               surf_map=roi,
                               bg_map=bg_map,
                               hemi=hemi,
                               view=view,
                               engine=engine,
                               avg_method=avg_method,
                               threshold=threshold,
                               cmap=cmap,
                               cbar_tick_format=cbar_tick_format,
                               alpha=alpha,
                               bg_on_data=bg_on_data,
                               vmin=vmin,
                               vmax=vmax,
                               title=title,
                               title_font_size=title_font_size,
                               output_file=output_file,
                               axes=axes,
                               figure=figure,
                               darkness=darkness,
                               categorical_cmap=categorical_cmap,
                               **kwargs)

    return display


def plot(args):
    results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution, args.mode))
    p_values_atlas_results_dir = str(os.path.join(results_path, "tmp", "p_values_atlas"))
    os.makedirs(p_values_atlas_results_dir, exist_ok=True)

    p_values_path = os.path.join(permutation_results_dir(args), f"p_values{get_hparam_suffix(args)}.p")
    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    rois_for_view = {
        "medial": ['G_precuneus', 'S_subparietal', 'G_cingul-Post-dorsal', 'S_parieto_occipital',
                   'G_oc-temp_med-Parahip', 'S_pericallosal'],
        "lateral": ['G_pariet_inf-Angular', 'G_occipital_middle', 'S_temporal_sup', 'S_front_inf',
                    'G_front_inf-Opercular', 'S_precentral-inf-part', 'G_orbital',
                    'G_pariet_inf-Supramar', 'G_temp_sup-Plan_tempo', 'S_interm_prim-Jensen', 'G_temp_sup-Lateral'], #, 'G_temporal_inf' , 'G_front_inf-Orbital'
        "ventral": ['S_oc-temp_lat', 'G_temporal_inf', 'G_orbital',
                    'Pole_temporal'] #'G_oc-temp_lat-fusifor', 'G_front_inf-Orbital'
    }

    unique_rois = set()
    for r in rois_for_view.values():
        unique_rois.update(r)

    label_names_dict = destrieux_label_names()
    color_palette = [(183, 242, 34), (127, 176, 4),
                     (174, 245, 176), (10, 250, 16), (4, 186, 8), (2, 110, 5), (1, 74, 3),
                     (193, 247, 233), (5, 245, 183), (1, 140, 104),
                     (145, 231, 242), (5, 220, 247), (0, 120, 135),
                     (115, 137, 245), (7, 48, 245), (2, 29, 158),
                     (174, 92, 237), (140, 7, 242), (76, 3, 133),
                     (245, 105, 242), (250, 5, 245), (125, 2, 122),
                     (242, 34, 152)]
    color_palette = [(x[0] / 255, x[1] / 255, x[2] / 255) for x in color_palette]

    assert len(unique_rois) <= len(color_palette), f"not enough colors for {len(unique_rois)} ROIS"

    all_colors = {label_names_dict[r]: c for r, c in
                  zip(unique_rois, color_palette)}

    save_legend(all_colors, p_values_atlas_results_dir)
    # plt.savefig(path)

    cbar_max = np.nanmax(np.concatenate((p_values['left'], p_values['right'])))
    cbar_min = 0
    for hemi in HEMIS:
        hemi_fs = FS_HEMI_NAMES[hemi]
        # resolution_fs = "fsaverage" if args.resolution == "fsaverage7" else args.resolution
        atlas_path = os.path.join(FREESURFER_HOME_DIR, f"subjects/{args.resolution}/label/{hemi_fs}.aparc.a2009s.annot")
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

        for i, (view, rois) in enumerate(rois_for_view.items()):
            regions_indices = [names.index(roi) for roi in rois]
            label_names = [label_names_dict[r] if r in label_names_dict else r.replace("-", " ") for r in rois]
            # colors = [atlas_colors[i][:4] / 255 for i in regions_indices]
            # colors = [(r, g, b, 0.5) for (r, g, b, a) in colors]
            # colors = [all_colors[l] for l in label_names]

            colors = [all_colors[label_names[regions_indices.index(i)]] if i in regions_indices else (0, 0, 0, 0) for i
                      in range(np.nanmax(atlas_labels))]
            cmap = ListedColormap(colors)
            atlas_labels_current_view = np.array([l if l in regions_indices else np.nan for l in atlas_labels])
            fig = plot_surf_roi_custom(
                fsaverage[f"infl_{hemi}"], roi_map=atlas_labels_current_view,
                bg_map=fsaverage[f"sulc_{hemi}"], hemi=hemi,
                view=view, alpha=0.5, cmap=cmap,
                bg_on_data=True, darkness=0.4, categorical_cmap=True,
            )

            plot_surf_stat_map_custom(
                fsaverage[f"infl_{hemi}"],
                p_values[hemi],
                hemi=hemi,
                view=view,
                colorbar=False,
                threshold=-np.log10(P_VALUE_THRESHOLD),
                vmax=cbar_max,
                vmin=cbar_min,
                cmap=CMAP_POS_ONLY,
                figure=fig,
                keep_bg=True,
            )

            title = f"{view}_{hemi}"
            path = os.path.join(p_values_atlas_results_dir, f"{title}.png")
            save_plot_and_crop_img(path)
            print(f"saved {path}")

    # plot for cbar:
    plotting.plot_surf_stat_map(
        fsaverage[f"infl_{HEMIS[0]}"],
        p_values[HEMIS[0]],
        hemi=HEMIS[0],
        view=args.views[0],
        colorbar=True,
        threshold=-np.log10(P_VALUE_THRESHOLD),
        vmax=cbar_max,
        vmin=cbar_min,
        cmap=CMAP_POS_ONLY,
    )
    save_plot_and_crop_img(os.path.join(p_values_atlas_results_dir, "colorbar.png"), crop_cbar=True)


def create_composite_image(args):
    results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution, args.mode))
    p_values_imgs_dir = str(os.path.join(results_path, "tmp", "p_values_atlas"))

    images_lateral = [Image.open(os.path.join(p_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["lateral"] for hemi
                      in HEMIS]
    images_medial = [Image.open(os.path.join(p_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["medial"] for hemi
                     in HEMIS]

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


    # without atlas
    results_path = str(os.path.join(RESULTS_DIR, "searchlight", args.model, args.features, args.resolution, args.mode))
    p_values_imgs_dir = str(os.path.join(results_path, "tmp", "p_values"))

    images_lateral = [Image.open(os.path.join(p_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["lateral"] for hemi
                      in HEMIS]
    images_medial = [Image.open(os.path.join(p_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["medial"] for hemi
                     in HEMIS]

    imgs_ventral = [Image.open(os.path.join(p_values_imgs_dir, f"ventral_{hemi}.png")) for hemi in HEMIS]
    img_ventral = append_images(images=imgs_ventral, horizontally=False)

    img_medial = append_images(images=images_medial)

    img_row_2 = append_images(images=[img_medial, img_ventral])

    img_colorbar = Image.open(os.path.join(p_values_imgs_dir, "colorbar.png"))
    img_lateral = append_images(images=images_lateral)

    img_row_1 = append_images([img_lateral, img_colorbar], padding=20)

    p_val_image = append_images([img_row_1, img_row_2], padding=5, horizontally=False)

    path = os.path.join(results_path, "searchlight_results_no_atlas.png")
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

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
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
