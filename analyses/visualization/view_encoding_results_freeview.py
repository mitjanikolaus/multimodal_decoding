import argparse
import os

from analyses.encoding.encoding_permutation_testing import permutation_results_dir, get_hparam_suffix, T_VAL_METRICS
from data import SELECT_DEFAULT, FEATURE_COMBINATION_CHOICES, VISION_FEATS_ONLY, LANG_FEATS_ONLY
from eval import METRIC_CROSS_ENCODING
from utils import ROOT_DIR, FREESURFER_HOME_DIR, HEMIS_FS, DEFAULT_RESOLUTION, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, \
    DEFAULT_MODEL


def run(args):
    os.environ["FREESURFER_HOME"] = FREESURFER_HOME_DIR
    os.system("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    cmd = "freeview"

    results_dir = permutation_results_dir(args)
    for hemi_fs in HEMIS_FS:
        cmd += f" -f $FREESURFER_HOME/subjects/fsaverage/surf/{hemi_fs}.inflated"

        for metric in [METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, METRIC_CROSS_ENCODING]:
            args.metric = metric
            mask_path = os.path.join(results_dir, "results_maps", f"tfce_values{get_hparam_suffix(args)}_{hemi_fs}.gii")

            # if metric == METRIC_CROSS_ENCODING:
            #     clusters_dir = os.path.join(results_dir, "results_maps", f"clusters{get_hparam_suffix(args)}")
            #     for file in glob.glob(clusters_dir + f"/{hemi_fs}*"):
            #         mask_paths.append(file)
            if os.path.isfile(mask_path):
                cmd += f":overlay={mask_path}:overlay_zorder=2"
            else:
                print(f'missing file: {mask_path}')

        for metric in T_VAL_METRICS:
            maps_path = os.path.join(results_dir, "results_maps", f"t_values_{metric}_{hemi_fs}.gii")
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

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--mod-specific-images-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-images-features", type=str, default=VISION_FEATS_ONLY)
    parser.add_argument("--mod-specific-images-test-features", type=str, default=VISION_FEATS_ONLY)

    parser.add_argument("--mod-specific-captions-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-captions-features", type=str, default=LANG_FEATS_ONLY)
    parser.add_argument("--mod-specific-captions-test-features", type=str, default=LANG_FEATS_ONLY)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)
    parser.add_argument("--tfce-dh", type=float, default=0.01)
    parser.add_argument("--tfce-clip", type=float, default=100)

    parser.add_argument("--n-clusters", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
