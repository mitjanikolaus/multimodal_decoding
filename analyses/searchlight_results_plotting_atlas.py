import argparse

import numpy as np
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import os
import pickle

import seaborn as sns

from analyses.ridge_regression_decoding import FEATS_SELECT_DEFAULT, get_default_features, FEATURE_COMBINATION_CHOICES
from analyses.searchlight import SEARCHLIGHT_OUT_DIR
from analyses.searchlight_permutation_testing import METRIC_CODES, METRIC_MIN
from analyses.searchlight_results_plotting import CMAP_POS_ONLY, DEFAULT_VIEWS
from utils import RESULTS_DIR, HEMIS

HCP_ATLAS_DIR = os.path.join("atlas_data", "hcp_surface")
HCP_ATLAS_LH = os.path.join(HCP_ATLAS_DIR, "lh.HCP-MMP1.annot")
HCP_ATLAS_RH = os.path.join(HCP_ATLAS_DIR, "rh.HCP-MMP1.annot")


def plot(args):
    results_path = os.path.join(RESULTS_DIR, "searchlight", args.resolution, args.features)

    p_values_path = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
        f"p_values_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
    )
    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    p_values['left'][p_values['left'] == 0] = np.nan
    p_values['right'][p_values['right'] == 0] = np.nan
    p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()

    regions_dicts = [
        {
            b'G_pariet_inf-Angular': 'Angular gyrus',
            b'G_precuneus': 'Precuneus',
            b'G_cingul-Post-dorsal': 'Posterior-dorsal part of the cingulate gyrus (dPCC)',
            b'G_pariet_inf-Supramar': 'Supramarginal gyrus',
            b'S_subparietal': 'Subparietal sulcus',
        },
        {
            b'Pole_temporal': 'Temporal pole',
            b'G_front_inf-Triangul': 'Triangular part of the inferior frontal gyrus',
            b'G_front_middle': 'Middle frontal gyrus',
            b'S_circular_insula_inf': 'Inferior segment of the circular sulcus of the insula',
        },
        # {
        #     b'G_temporal_middle': 'Middle temporal gyrus',
        #     b'G_temporal_inf': 'Inferior temporal gyrus',
        #     b'G_orbital': 'Orbital gyri',
        # }
    ]

    # not/less relevant:
    # b'S_circular_insula_sup': 'Superior segment of the circular sulcus of the insula',
    # b'G_Ins_lg_and_S_cent_ins': 'Long insular gyrus and central sulcus of the insula',
    # b'G_temp_sup-G_T_transv': 'Anterior transverse temporal gyrus (of Heschl)',
    # b'G_temp_sup-Lateral': 'Lateral aspect of the superior temporal gyrus',
    # b'G_front_inf-Opercular': 'Opercular part of the inferior frontal gyrus',
    # b'S_collat_transv_ant': 'Anterior transverse collateral sulcus',
    # b'S_parieto_occipital': 'Parieto-occipital sulcus',
    # b'G_cuneus': 'Cuneus',
    # b'G_and_S_cingul-Mid-Post': 'Middle-posterior part of the cingulate gyrus and sulcus (pMCC)',
    # b'G_oc-temp_lat-fusifor': 'Lateral occipito-temporal gyrus (fusiform gyrus)',
    # b'G_and_S_cingul-Mid-Ant': 'Middle-anterior part of the cingulate gyrus and sulcus (aMCC)',
    # b'G_and_S_cingul-Ant': 'Anterior part of the cingulate gyrus and sulcus (ACC)',
    # b'G_occipital_middle': 'Middle occipital gyrus',
    # b'G_and_S_frontomargin': 'Fronto-marginal gyrus (of Wernicke) and sulcus',

    for r, regions_dict in enumerate(regions_dicts):
        # get indices in atlas for these labels
        regions_indices = [
            np.where(np.array(destrieux_atlas['labels']) == region)[0][0]
            for region in regions_dict
        ]

        labels = list(regions_dict.values())
        colors = sns.color_palette("Set2", n_colors=len(labels) + 1)[1:]

        fig = plt.figure(figsize=(5 * len(args.views), 2))
        # fig.suptitle(f'{args.metric}: -log10(p_value)', x=0, horizontalalignment="left")
        axes = fig.subplots(nrows=1, ncols=2 * len(args.views), subplot_kw={'projection': '3d'})
        cbar_max = np.nanmax(np.concatenate((p_values['left'], p_values['right'])))
        cbar_min = 0
        for i, view in enumerate(args.views):
            for j, hemi in enumerate(HEMIS):
                scores_hemi = p_values[hemi]
                infl_mesh = fsaverage[f"infl_{hemi}"]
                fig_hemi = plotting.plot_surf_stat_map(
                    infl_mesh,
                    scores_hemi,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    axes=axes[i * 2 + j],
                    colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                    threshold=1.3,  # -log10(0.05) ~ 1.3
                    vmax=cbar_max,
                    vmin=cbar_min,
                    cmap=CMAP_POS_ONLY,
                    symmetric_cbar=False,
                )

                parcellation = destrieux_atlas[f'map_{hemi}']
                plotting.plot_surf_contours(infl_mesh, parcellation, labels=labels,
                                            levels=regions_indices, figure=fig_hemi, axes=axes[i * 2 + j],
                                            legend=False,
                                            colors=colors)

                axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)

        title = f"{args.model}_{args.mode}_metric_{METRIC_CODES[args.metric]}_p_values_atlas_{r}"
        fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
        results_searchlight = os.path.join(results_path, f"{title}.png")
        plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
        plt.close()


    # from nibabel.gifti import GiftiDataArray, GiftiImage
    # for hemi in HEMIS:
    #     data = t_values[hemi][METRIC_MIN].astype(np.float32)
    #     # gimage = GiftiImage(darrays=[GiftiDataArray(data, intent='z score', datatype="float32")])
    #     # img_surf = nibabel.freesurfer.mghformat.load()
    #     infl_mesh = fsaverage[f"infl_{hemi}"]
    #     niimg = datasets.load_mni152_template()
    #
    #     gimage = MGHImage(data, affine=niimg.affine)
    #     # nifti1.data_type_codes
    #     # gimage.add_gifti_data_array([GiftiDataArray(data, intent='z score', datatype="float32")])
    #
    #     nibabel.save(gimage, f"t_values_{hemi}.nii.gz")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='blip2')
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default='fsaverage5')
    parser.add_argument("--mode", type=str, default='n_neighbors_100')

    parser.add_argument("--smoothing-iterations", type=int, default=0)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--metric", type=str, default=METRIC_MIN)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.features = get_default_features(args.model) if args.features == FEATS_SELECT_DEFAULT else args.features

    if args.resolution != 'fsaverage5':
        raise NotImplementedError()
