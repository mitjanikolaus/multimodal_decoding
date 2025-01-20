import argparse
import os

from analyses.encoding.encoding_permutation_testing import permutation_results_dir, get_hparam_suffix, \
    CORR_IMAGES_MOD_SPECIFIC_IMAGES, CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS
from analyses.encoding.ridge_regression_encoding import ENCODING_RESULTS_DIR
from data import SELECT_DEFAULT, FEATURE_COMBINATION_CHOICES
from eval import CORR_CAPTIONS, CORR_IMAGES, CORR_CROSS_IMAGES_TO_CAPTIONS, CORR_CROSS_CAPTIONS_TO_IMAGES, \
    METRIC_CROSS_ENCODING
from utils import ROOT_DIR, FREESURFER_HOME_DIR, HEMIS_FS, DEFAULT_RESOLUTION, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC

METRICS = [CORR_CAPTIONS, CORR_IMAGES, CORR_CROSS_IMAGES_TO_CAPTIONS, CORR_CROSS_CAPTIONS_TO_IMAGES,
           CORR_IMAGES_MOD_SPECIFIC_IMAGES, CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS]


def run(args):
    os.environ["FREESURFER_HOME"] = FREESURFER_HOME_DIR
    os.system("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    cmd = "freeview"

    for hemi_fs in HEMIS_FS:
        cmd += f" -f $FREESURFER_HOME/subjects/fsaverage/surf/{hemi_fs}.inflated"

        results_dir = permutation_results_dir(args)
        mask_paths = []
        for metric in [METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_ENCODING]:
            args.metric = metric
            mask_paths.append(
                os.path.join(results_dir, "results_maps", f"tfce_values{get_hparam_suffix(args)}_{hemi_fs}.gii"))

            # if metric == METRIC_CROSS_ENCODING:
            #     clusters_dir = os.path.join(results_dir, "results_maps", f"clusters{get_hparam_suffix(args)}")
            #     for file in glob.glob(clusters_dir + f"/{hemi_fs}*"):
            #         mask_paths.append(file)

        for mask_path in mask_paths:
            if os.path.isfile(mask_path):
                cmd += f":overlay={mask_path}:overlay_zorder=2"

        maps_paths = [os.path.join(ENCODING_RESULTS_DIR, "corr_results_maps", f"{metric}_{hemi_fs}.gii") for metric in METRICS]
        for maps_path in maps_paths:
            if os.path.isfile(maps_path):
                cmd += f":overlay={maps_path}:overlay_zorder=2"

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
    parser.add_argument("--p-values-threshold", type=float, default=0.05)

    parser.add_argument("--model", type=str, default='imagebind')
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--mod-specific-vision-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-vision-features", type=str, default=SELECT_DEFAULT)
    parser.add_argument("--mod-specific-vision-test-features", type=str, default=SELECT_DEFAULT)

    parser.add_argument("--mod-specific-lang-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-lang-features", type=str, default=SELECT_DEFAULT)
    parser.add_argument("--mod-specific-lang-test-features", type=str, default=SELECT_DEFAULT)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)
    parser.add_argument("--tfce-dh", type=float, default=0.1)
    parser.add_argument("--tfce-clip", type=float, default=100)

    parser.add_argument("--n-clusters", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
