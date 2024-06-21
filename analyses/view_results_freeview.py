
import argparse
import os

from utils import SUBJECTS, FREESURFER_BASE_DIR, FMRI_RAW_DATA_DIR, FMRI_PREPROCESSED_DATA_DIR

# freeview \
# -f $FREESURFER_HOME/subjects/fsaverage/surf/lh.inflated\
# :overlay=~/data/multimodal_decoding/searchlight/train/blip2/avg/fsaverage7/n_neighbors_200/p_values_gifti/lh.gii\
# :annot=~/code/multimodal_decoding/atlas_data/hcp_surface/lh.HCP-MMP1.annot\
# :annot=$FREESURFER_HOME/subjects/fsaverage/label/lh.aparc.annot\
# :annot=$FREESURFER_HOME/subjects/fsaverage/label/lh.aparc.a2009s.annot \
# -f $FREESURFER_HOME/subjects/fsaverage/surf/rh.inflated\
# :overlay=~/data/multimodal_decoding/searchlight/train/blip2/avg/fsaverage7/n_neighbors_200/p_values_gifti/rh.gii\
# :annot=~/code/multimodal_decoding/atlas_data/hcp_surface/rh.HCP-MMP1.annot\
# :annot=$FREESURFER_HOME/subjects/fsaverage/label/rh.aparc.annot\
# :annot=$FREESURFER_HOME/subjects/fsaverage/label/rh.aparc.a2009s.annot


def run(args):
    os.environ["FREESURFER_HOME"] = f"/usr/local/freesurfer/7.4.1"
    result_code = os.system("$FREESURFER_HOME/SetUpFreeSurfer.sh")
    cmd = (f"freeview -f $FREESURFER_HOME/subjects/fsaverage/surf/lh.inflated")

    mask_paths = [os.path.expanduser("~/data/multimodal_decoding/searchlight/train/blip2/avg/fsaverage7/n_neighbors_200/p_values_gifti/lh.gii")]
    mask_paths += [os.path.expanduser(f"~/data/multimodal_decoding/searchlight/train/blip2/avg/fsaverage7/n_neighbors_200/p_values_gifti/thresh_0.01_lh_cluster_{i}.gii") for i in range(10)]
    for mask_path in mask_paths:
        cmd += f":overlay={mask_path}"
    annot_path = os.path.expanduser("~/code/multimodal_decoding/atlas_data/hcp_surface/lh.HCP-MMP1.annot")
    cmd += f":annot={annot_path}"
    result_code = os.system(cmd)
    if result_code != 0:
        raise RuntimeError(f"failed to start freeview with error code {result_code}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--resolution", type=str, default="fsaverage7")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
