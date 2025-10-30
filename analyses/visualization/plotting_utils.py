from warnings import warn

import numpy as np
from PIL import Image
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import make_axes
from matplotlib.colors import Normalize, to_rgba
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from nilearn.plotting._utils import get_colorbar_and_data_ranges
from nilearn.plotting.cm import mix_colormaps
from nilearn.plotting.surface._matplotlib_backend import _get_view_plot_surf, _compute_facecolors, \
    _compute_surf_map_faces, _threshold_and_rescale, _get_ticks, _get_cmap
from nilearn.plotting.surface._utils import get_faces_on_edge
from nilearn.surface import load_surf_mesh
from nilearn.surface.surface import check_extensions, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS, load_surf_data

from data import TEST_IMAGES, TEST_CAPTIONS, clean_metric_name
from utils import DIFF, DIFF_DECODERS

CBAR_T_VAL_MAX = 15


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
    elev, azim = _get_view_plot_surf(hemi, view)

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
        bg_face_colors = _compute_facecolors(
            bg_map, faces, coords.shape[0], darkness, alpha
        )
    if surf_map is not None:
        surf_map_faces = _compute_surf_map_faces(
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
            ticks = _get_ticks(cbar_vmin, cbar_vmax,
                               cbar_tick_format, threshold)
            our_cmap, norm = _get_cmap(cmap,
                                       threshold,
                                       vmax,
                                       cbar_tick_format,
                                       threshold)
            if '$' in metric:
                ticks = [threshold, round(np.mean([threshold, cbar_vmax]), 1), cbar_vmax]
                if metric.split('$')[0] == DIFF:
                    _, training_mode, metric_1, metric_2 = metric.split('$')
                    metric_1 = clean_metric_name(metric_1)
                    metric_2 = clean_metric_name(metric_2)
                    label = f"{training_mode} decoder | {metric_1} - {metric_2}"
                elif metric.split('$')[0] == DIFF_DECODERS:
                    _, training_mode_1, training_mode_2, metric_name = metric.split('$')
                    metric_name = clean_metric_name(metric_name)
                    label = f"{training_mode_1} decoder - {training_mode_2} decoder | {metric_name}"
                else:
                    training_mode, metric_name = metric.split('$')
                    metric_name = clean_metric_name(metric_name)
                    label = f"{training_mode} decoder | {metric_name}"

            # elif metric.startswith("pairwise_acc"):
            #     # ticks = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
            #     # cbar_vmin = 0.5
            #     cbar_vmin = 0
            #     # cbar_vmax = CBAR_T_VAL_MAX
            #     ticks = [threshold, round(np.mean([threshold, cbar_vmax]), 1), cbar_vmax]
            #     label = metric.replace("pairwise_acc_", "")
            else:
                ticks = [threshold, round(np.mean([threshold, np.max(ticks)]), -4), int(np.max(ticks) / 1000) * 1000]
                # ticks = [threshold, np.mean([threshold, np.max(ticks)]), np.max(ticks)]
                label = f"log(TFCE)"
                # cbar_vmin = 0

            bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)
            # we need to create a proxy mappable
            proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
            proxy_mappable.set_array(surf_map_faces)
            cax, _ = make_axes(axes, location='bottom', fraction=.15,
                               shrink=.5, pad=.0, aspect=10.)

            # else:
            #     ticks = [0.5, 0.6, threshold, 0.7, 0.8, 0.9]
            figure.colorbar(
                proxy_mappable, cax=cax, ticks=ticks, label=label,
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
        faces_outside = get_faces_on_edge(faces, roi_indices)
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
