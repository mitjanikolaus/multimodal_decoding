import argparse
import os

from analyses.ridge_regression_decoding import get_default_features, FEATS_SELECT_DEFAULT, FEATURE_COMBINATION_CHOICES
from analyses.searchlight import SEARCHLIGHT_OUT_DIR
from utils import ROOT_DIR, FREESURFER_HOME_DIR, HEMIS_FS


def run(args):
    os.environ["FREESURFER_HOME"] = FREESURFER_HOME_DIR
    os.system("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    cmd = (f"freeview")
    thresh = str(args.p_values_threshold)
    for hemi_fs in HEMIS_FS:
        cmd += f" -f $FREESURFER_HOME/subjects/fsaverage/surf/{hemi_fs}.inflated"
        mask_paths = [os.path.join(
            SEARCHLIGHT_OUT_DIR,
            f"train/{args.model}/{args.features}/fsaverage7/{args.mode}/p_values_gifti/{hemi_fs}.gii")
        ]
        mask_paths += [os.path.join(
            SEARCHLIGHT_OUT_DIR,
            f"train/{args.model}/{args.features}/fsaverage7/{args.mode}/p_values_gifti/thresh_{thresh}_{hemi_fs}"
            f"_cluster_{i}.gii")
            for i in range(args.n_clusters)
        ]
        for mask_path in mask_paths:
            cmd += f":overlay={mask_path}:overlay_zorder=2"
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

    parser.add_argument("--p-values-threshold", type=float, default=0.01)
    parser.add_argument("--model", type=str, default='blip2')
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--mode", type=str, default='n_neighbors_200')

    parser.add_argument("--n-clusters", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.features = get_default_features(args.model) if args.features == FEATS_SELECT_DEFAULT else args.features

    run(args)
