import argparse
import glob
import os

from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, get_hparam_suffix, \
    add_searchlight_permutation_args, T_VAL_METRICS
from eval import ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, ACC_IMAGERY_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_IMAGES, ACC_IMAGERY_NO_STD_MOD_SPECIFIC_IMAGES, \
    ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_IMAGES, ACC_IMAGERY_MOD_SPECIFIC_CAPTIONS, \
    ACC_IMAGERY_NO_STD_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_CAPTIONS, \
    ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_CAPTIONS, ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC
from utils import ROOT_DIR, FREESURFER_HOME_DIR, HEMIS_FS, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_DECODING, \
    METRIC_MOD_AGNOSTIC_AND_CROSS

METRICS = T_VAL_METRICS + [METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_DECODING,
                           METRIC_MOD_AGNOSTIC_AND_CROSS] + [ACC_IMAGERY_MOD_SPECIFIC_IMAGES,
                                                             ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_IMAGES,
                                                             ACC_IMAGERY_NO_STD_MOD_SPECIFIC_IMAGES,
                                                             ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_IMAGES,
                                                             ACC_IMAGERY_MOD_SPECIFIC_CAPTIONS,
                                                             ACC_IMAGERY_WHOLE_TEST_SET_MOD_SPECIFIC_CAPTIONS,
                                                             ACC_IMAGERY_NO_STD_MOD_SPECIFIC_CAPTIONS,
                                                             ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_SPECIFIC_CAPTIONS]


def run(args):
    os.environ["FREESURFER_HOME"] = FREESURFER_HOME_DIR
    os.system("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    cmd = "freeview"

    for hemi_fs in HEMIS_FS:
        cmd += f" -f $FREESURFER_HOME/subjects/fsaverage/surf/{hemi_fs}.inflated"

        results_dir = permutation_results_dir(args)
        mask_paths = []
        for metric in [METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_DECODING, METRIC_MOD_AGNOSTIC_AND_CROSS,
                       ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC, ACC_IMAGERY]:
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

        maps_paths = [os.path.join(results_dir, "acc_results_maps", f"{metric}_{hemi_fs}.gii") for metric in METRICS]
        for maps_path in maps_paths:
            if os.path.isfile(maps_path):
                cmd += f":overlay={maps_path}:overlay_zorder=2"
            else:
                print(f"missing acc result map: {maps_path}")

        annot_paths = [os.path.join(FREESURFER_HOME_DIR, f"subjects/fsaverage/label/{hemi_fs}.{atlas_name}") for
                       atlas_name in ["aparc.annot", "aparc.a2009s.annot"]]
        annot_paths += [os.path.join(ROOT_DIR, f"atlas_data/hcp_surface/{hemi_fs}.HCP-MMP1.annot")]
        for annot_path in annot_paths:
            cmd += f":annot={annot_path}:annot_zorder=1"

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
