import argparse
import glob
import os

from scipy.stats import pearsonr
import nibabel as nib
from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, get_hparam_suffix, \
    add_searchlight_permutation_args
from data import IMAGE, CAPTION
from eval import ACC_IMAGERY
from utils import ROOT_DIR, FREESURFER_HOME_DIR, HEMIS_FS, METRIC_CROSS_DECODING, \
    METRIC_MOD_AGNOSTIC_AND_CROSS, METRIC_DIFF_ATTENTION


def run(args):
    os.environ["FREESURFER_HOME"] = FREESURFER_HOME_DIR
    os.system("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    cmd = "freeview"

    for hemi_fs in HEMIS_FS:
        cmd += f" -f $FREESURFER_HOME/subjects/fsaverage/surf/{hemi_fs}.inflated"

        results_dir = permutation_results_dir(args)
        mask_paths = []
        for metric in [METRIC_MOD_AGNOSTIC_AND_CROSS, METRIC_DIFF_ATTENTION]: #, METRIC_CROSS_DECODING, ACC_IMAGERY]:
            args.metric = metric
            mask_paths.append(
                os.path.join(results_dir, "results_maps", f"tfce_values{get_hparam_suffix(args)}_{hemi_fs}.gii"))

            if metric == METRIC_MOD_AGNOSTIC_AND_CROSS:
                clusters_dir = os.path.join(results_dir, "results_maps", f"clusters{get_hparam_suffix(args)}")
                for file in glob.glob(clusters_dir + f"/{hemi_fs}*"):
                    mask_paths.append(file)

        for mask_path in mask_paths:
            if os.path.isfile(mask_path):
                cmd += f":overlay={mask_path}:overlay_zorder=2"
            else:
                print(f"missing mask: {mask_path}")

        maps_paths = glob.glob(os.path.join(results_dir, "acc_results_maps", f"*_{hemi_fs}.gii"))
        for maps_path in maps_paths:
            if 'diff' in maps_path:
                low = 0.05
                high = 0.1
            else:
                if 'test_caption' in maps_path:
                    low = 0.55
                    high = 0.65
                elif 'test_image' in maps_path:
                    low = 0.55
                    high = 0.8
                else:
                    low = 0.55
                    high = 0.7

            cmd += f":overlay={maps_path}:overlay_zorder=2:overlay_threshold={low},{high}"

        annot_paths = [os.path.join(FREESURFER_HOME_DIR, f"subjects/fsaverage/label/{hemi_fs}.{atlas_name}") for
                       atlas_name in ["aparc.annot", "aparc.a2009s.annot"]]
        annot_paths += [os.path.join(ROOT_DIR, f"atlas_data/hcp_surface/{hemi_fs}.HCP-MMP1.annot")]
        for annot_path in annot_paths:
            cmd += f":annot={annot_path}:annot_zorder=1"

        results = {}
        for modality in [IMAGE, CAPTION]:
            for attention in ["attended", "unattended"]:
                path = os.path.join(results_dir, "acc_results_maps", f'agnostic_decoder_test_{modality}_{attention}_{hemi_fs}.gii')
                data = nib.load(path)
                results[f"test_{modality}_{attention}"] = data.darrays[0].data

        corr_attended = pearsonr(results['test_image_attended'], results['test_caption_attended'])
        print(f'{hemi_fs} corr_attended: {corr_attended[0]:.2f}')
        corr_unattended = pearsonr(results['test_image_unattended'], results['test_caption_unattended'])
        print(f'{hemi_fs} corr_unattended: {corr_unattended[0]:.2f}')

        corr_image = pearsonr(results['test_image_attended'], results['test_image_unattended'])
        print(f'{hemi_fs} corr_image: {corr_image[0]:.2f}')

        corr_caption = pearsonr(results['test_caption_attended'], results['test_caption_unattended'])
        print(f'{hemi_fs} corr_caption: {corr_caption[0]:.2f}')



    result_code = os.system(cmd)
    if result_code != 0:
        raise RuntimeError(f"failed to start freeview with error code {result_code}")


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    parser.add_argument("--p-values-threshold", type=float, default=0.05)

    parser.add_argument("--n-clusters", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
