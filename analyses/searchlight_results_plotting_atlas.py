import argparse
import warnings

import nibabel
import numpy as np
from nibabel import nifti1, MGHImage
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
import pickle

from nilearn.surface import load_surf_data
from tqdm import tqdm

from analyses.ridge_regression_decoding import FEATS_SELECT_DEFAULT, get_default_features, FEATURE_COMBINATION_CHOICES
from analyses.searchlight import SEARCHLIGHT_OUT_DIR
from analyses.searchlight_permutation_testing import METRIC_MIN_DIFF_BOTH_MODALITIES, METRIC_DIFF_IMAGES, \
    METRIC_DIFF_CAPTIONS, METRIC_CAPTIONS, METRIC_IMAGES, load_per_subject_scores, CHANCE_VALUES
from utils import RESULTS_DIR, SUBJECTS, HEMIS

VIEWS = ["lateral", "medial", "ventral", "posterior"]

DEFAULT_T_VALUE_THRESH = 0.2
DEFAULT_TFCE_VAL_THRESH = 50


def plot_test_statistics(test_statistics, args, filename_suffix=""):
    print(f"plotting test stats {filename_suffix}")
    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
    metric = METRIC_MIN_DIFF_BOTH_MODALITIES
    fig = plt.figure(figsize=(5 * len(VIEWS), len(test_statistics) * 2))
    subfigs = fig.subfigures(nrows=len(test_statistics), ncols=1)
    cbar_max = {stat: None for stat in test_statistics.keys()}
    for subfig, (stat_name, values) in zip(subfigs, test_statistics.items()):
        # subfig.suptitle(f'{metric} {stat_name}', x=0, horizontalalignment="left")
        axes = subfig.subplots(nrows=1, ncols=2 * len(VIEWS), subplot_kw={'projection': '3d'})
        for i, view in enumerate(VIEWS):
            for j, hemi in enumerate(HEMIS):
                scores_hemi = values[hemi][metric]
                infl_mesh = fsaverage[f"infl_{hemi}"]
                if cbar_max[stat_name] is None:
                    if (stat_name == "t-values-smoothed") and (cbar_max['t-values'] is not None):
                        cbar_max[stat_name] = cbar_max['t-values']
                    else:
                        cbar_max[stat_name] = np.nanmax(scores_hemi)
                threshold = DEFAULT_T_VALUE_THRESH if stat_name.startswith("t-values") else DEFAULT_TFCE_VAL_THRESH
                plotting.plot_surf_stat_map(
                    infl_mesh,
                    scores_hemi,
                    hemi=hemi,
                    view=view,
                    bg_map=fsaverage[f"sulc_{hemi}"],
                    bg_on_data=True,
                    axes=axes[i * 2 + j],
                    colorbar=True if axes[i * 2 + j] == axes[-1] else False,
                    threshold=threshold,
                    vmax=cbar_max[stat_name],
                    vmin=0,
                    cmap="hot",
                )
                axes[i * 2 + j].set_title(f"{hemi} {view}", y=0.85, fontsize=10)

                destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
                parcellation = destrieux_atlas[f'map_{hemi}']

                # regions_dict = {b'L G_pariet_inf-Angular': 'left angular gyrus'}
                regions_dict = {b'G_postcentral': 'Postcentral gyrus',
                                b'G_precentral': 'Precentral gyrus'}

                # get indices in atlas for these labels
                # regions_indices = [
                #     [i for i, l in destrieux_atlas['labels'] if l == region][0]
                #     for region in regions_dict
                # ]
                regions_indices = [
                    np.where(np.array(destrieux_atlas['labels']) == region)[0][0]
                    for region in regions_dict
                ]
                colors = ['g', 'b']
                labels = list(regions_dict.values())
                plotting.plot_surf_contours(infl_mesh, parcellation, labels=labels,
                                            levels=regions_indices, axes=axes[i * 2 + j],
                                            legend=True,
                                            colors=colors)

    title = f"{args.model}_{args.mode}_test_stats{filename_suffix}_rois"
    fig.subplots_adjust(left=0, right=0.85, bottom=0, wspace=-0.1, hspace=0, top=1)
    results_searchlight = os.path.join(RESULTS_DIR, "searchlight", args.resolution, f"{title}.png")
    os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
    plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
    plt.close()


HCP_ATLAS_DIR = os.path.join("atlas_data", "hcp_surface")
HCP_ATLAS_LH = os.path.join(HCP_ATLAS_DIR, "lh.HCP-MMP1.annot")
HCP_ATLAS_RH = os.path.join(HCP_ATLAS_DIR, "rh.HCP-MMP1.annot")


def run(args):
    # atlas_data = load_surf_data(HCP_ATLAS_LH)


    t_values_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
                                 "t_values.p")
    t_values = pickle.load(open(t_values_path, 'rb'))

    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)

    from nibabel.gifti import GiftiDataArray, GiftiImage
    for hemi in HEMIS:
        data = t_values[hemi][METRIC_MIN_DIFF_BOTH_MODALITIES].astype(np.float32)
        # gimage = GiftiImage(darrays=[GiftiDataArray(data, intent='z score', datatype="float32")])
        # img_surf = nibabel.freesurfer.mghformat.load()
        infl_mesh = fsaverage[f"infl_{hemi}"]
        niimg = datasets.load_mni152_template()

        gimage = MGHImage(data, affine=niimg.affine)
        # nifti1.data_type_codes
        # gimage.add_gifti_data_array([GiftiDataArray(data, intent='z score', datatype="float32")])

        nibabel.save(gimage, f"t_values_{hemi}.nii.gz")
        # gimage.to_filename(f"t_values_{hemi}.gii")
        # write(gimage, some_path)



    t_values_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
                                 "t_values.p")
    t_values = pickle.load(open(t_values_path, 'rb'))
    t_values_smooth_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution,
                                        args.mode,
                                        f"t_values_smoothed_{args.smoothing_iterations}.p")
    t_values_smooth = pickle.load(open(t_values_smooth_path, 'rb'))
    # tfce_values_path = os.path.join(SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
    #                              f"tfce_values_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p")
    # tfce_values = pickle.load(open(tfce_values_path, 'rb'))
    test_statistics = {"t-values": t_values, "t-values-smoothed": t_values_smooth}#, "tfce-values": tfce_values}
    plot_test_statistics(test_statistics, args)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='vilt')
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default='fsaverage5')
    parser.add_argument("--mode", type=str, default='n_neighbors_100')

    parser.add_argument("--smoothing-iterations", type=int, default=0)

    parser.add_argument("--tfce", action="store_true")
    parser.add_argument("--tfce-h", type=float, default=2)
    parser.add_argument("--tfce-e", type=float, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.features = get_default_features(args.model) if args.features == FEATS_SELECT_DEFAULT else args.features

    run(args)
