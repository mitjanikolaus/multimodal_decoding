import argparse
from warnings import warn

import nibabel.freesurfer
import numpy as np
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import make_axes
from matplotlib.colors import Normalize, to_rgba
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import pickle

from nilearn.plotting.cm import mix_colormaps
from nilearn.plotting.img_plotting import get_colorbar_and_data_ranges
from nilearn.plotting.surf_plotting import _get_cmap_matplotlib, \
    _get_ticks_matplotlib, _threshold_and_rescale, _compute_surf_map_faces_matplotlib, _get_view_plot_surf_matplotlib, \
    _compute_facecolors_matplotlib, _get_faces_on_edge
from nilearn.surface import load_surf_mesh
from nilearn.surface.surface import check_extensions, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS, load_surf_data

from analyses.encoding_permutation_testing import permutation_results_dir, get_hparam_suffix
from analyses.searchlight.searchlight_permutation_testing import calc_significance_cutoff
from analyses.searchlight.searchlight_results_plotting import CMAP_POS_ONLY, DEFAULT_VIEWS, save_plot_and_crop_img, \
    append_images
from utils import RESULTS_DIR, HEMIS, FREESURFER_HOME_DIR, FS_HEMI_NAMES, DEFAULT_RESOLUTION, SUBJECTS, \
    METRIC_CROSS_ENCODING, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC

HCP_ATLAS_DIR = os.path.join("atlas_data", "hcp_surface")
HCP_ATLAS_LH = os.path.join(HCP_ATLAS_DIR, "lh.HCP-MMP1.annot")
HCP_ATLAS_RH = os.path.join(HCP_ATLAS_DIR, "rh.HCP-MMP1.annot")

CMAP_POS_ONLY = "hot"


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
                                 axes=None, figure=None, categorical_cmap=False, metric=None):
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
            cbar_vmax = cbar_vmax if cbar_vmax is not None else vmax
            ticks = _get_ticks_matplotlib(cbar_vmin, cbar_vmax,
                                          cbar_tick_format, threshold)
            our_cmap, norm = _get_cmap_matplotlib(cmap,
                                                  threshold,
                                                  vmax,
                                                  cbar_tick_format,
                                                  threshold)
            cbar_vmin = 0
            bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)
            # we need to create a proxy mappable
            proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(surf_map_faces)
            cax, _ = make_axes(axes, location='bottom', fraction=.15,
                               shrink=.5, pad=.0, aspect=10.)
            # if metric == "tfce":
            ticks = [0, threshold,  round(np.mean([threshold, np.max(ticks)]), -3), round(np.max(ticks), -3)]
            # else:
            #     ticks = [0.5, 0.6, threshold, 0.7, 0.8, 0.9]
            figure.colorbar(
                proxy_mappable, cax=cax, ticks=ticks, label="TFCE",
                boundaries=bounds, spacing='proportional',
                format=ScalarFormatter(useOffset=False), orientation='horizontal')
            cax.xaxis.set_ticks_position('top')

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
        figure=None, bg_map=None, bg_on_data=False, keep_bg=False, darkness=0.7, categorical_cmap=False, metric=None,
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
            output_file=output_file, axes=axes, figure=figure, categorical_cmap=categorical_cmap, metric=metric)

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
        bg_map=None, bg_on_data=False, keep_bg=False, categorical_cmap=False, metric=None, **kwargs
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
        cbar_vmax=cbar_vmax, categorical_cmap=categorical_cmap, metric=metric, **kwargs
    )
    return display


def plot_surf_contours_custom(surf_mesh, roi_map, axes=None, figure=None, levels=None,
                       labels=None, colors=None, legend=False, cmap='tab20',
                       title=None, output_file=None, **kwargs):
    """Plot contours of ROIs on a surface, \
    optionally over a statistical map.

    Parameters
    ----------
    surf_mesh : str or list of two numpy.ndarray
        Surface :term:`mesh` geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the :term:`mesh` :term:`vertices<vertex>`,
        the second containing the indices (into coords)
        of the :term:`mesh` :term:`faces`.

    roi_map : str or numpy.ndarray or list of numpy.ndarray
        ROI map to be displayed on the surface mesh,
        can be a file
        (valid formats are .gii, .mgz, or
        Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array with a value for each :term:`vertex` of the `surf_mesh`.
        The value at each :term:`vertex` one inside the ROI
        and zero inside ROI,
        or an integer giving the label number for atlases.

    axes : instance of matplotlib axes, None, optional
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, uses axes from figure if available, else creates new axes.
    %(figure)s
    levels : list of integers, or None, optional
        A list of indices of the regions that are to be outlined.
        Every index needs to correspond to one index in roi_map.
        If None, all regions in roi_map are used.

    labels : list of strings or None, or None, optional
        A list of labels for the individual regions of interest.
        Provide None as list entry to skip showing the label of that region.
        If None no labels are used.

    colors : list of matplotlib color names or RGBA values, or None, optional
        Colors to be used.

    legend : boolean,  optional, default=False
        Whether to plot a legend of region's labels.
    %(cmap)s
        Default='tab20'.
    %(title)s
    %(output_file)s

    kwargs: extra keyword arguments, optional
        Extra keyword arguments passed to
        :func:`~nilearn.plotting.plot_surf`.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    """
    if figure is None and axes is None:
        figure = plot_surf_custom(surf_mesh, **kwargs)
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
        _ = plot_surf_custom(surf_mesh, axes=axes, **kwargs)

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
            colors = [to_rgba(color, alpha=1.) for color in colors]
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
        faces_outside = _get_faces_on_edge(faces, roi_indices)
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
    plt.style.use("dark_background")

    for result_metric in [METRIC_CROSS_ENCODING]: #METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC
        results_path = str(os.path.join(RESULTS_DIR, "encoding", args.model, args.features, args.resolution))
        atlas_tmp_results_dir = str(os.path.join(results_path, "tmp", f"{result_metric}_atlas"))
        os.makedirs(atlas_tmp_results_dir, exist_ok=True)

        args.metric = result_metric
        tfce_values_null_distribution_path = os.path.join(
            permutation_results_dir(args), f"tfce_values_null_distribution{get_hparam_suffix(args)}.p"
        )
        null_distribution_tfce_values = pickle.load(open(tfce_values_null_distribution_path, "rb"))
        significance_cutoff, _ = calc_significance_cutoff(null_distribution_tfce_values, result_metric, args.p_value_threshold)

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
    for result_metric in [METRIC_CROSS_ENCODING]: #METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC
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

    parser.add_argument("--model", type=str, default='imagebind')
    parser.add_argument("--features", type=str, default="avg_test_avg")

    parser.add_argument("--mod-specific-vision-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-vision-features", type=str, default="vision_test_vision")

    parser.add_argument("--mod-specific-lang-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-lang-features", type=str, default="lang_test_lang")

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)
    parser.add_argument("--tfce-dh", type=float, default=0.01)
    parser.add_argument("--tfce-clip", type=float, default=100)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)

    parser.add_argument("--p-value-threshold", type=float, default=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    plot(args)
    create_composite_image(args)
