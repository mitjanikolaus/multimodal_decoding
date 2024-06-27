import argparse
import copy

import numpy as np
from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from nibabel.nifti1 import intent_codes, data_type_codes
import os
import pickle

from analyses.ridge_regression_decoding import FEATS_SELECT_DEFAULT, get_default_features, FEATURE_COMBINATION_CHOICES
from analyses.searchlight.searchlight import SEARCHLIGHT_OUT_DIR
from analyses.searchlight.searchlight_permutation_testing import METRIC_CODES, METRIC_MIN, calc_clusters, \
    get_edge_lengths_dicts_based_on_edges
from analyses.searchlight.searchlight_results_masks import FS_HEMI_NAMES
from utils import HEMIS


def export_to_gifti(scores, path):
    data = scores.astype(np.float32)
    gimage = GiftiImage(
        darrays=[GiftiDataArray(
            data,
            intent=intent_codes.code['NIFTI_INTENT_NONE'],
            datatype=data_type_codes.code['NIFTI_TYPE_FLOAT32'])]
    )
    gimage.to_filename(path)


def create_masks(args):
    masks = []
    for path in args.paths:
        masks.append(pickle.load(open(path, "rb")))

    combined_mask = dict()
    for hemi in HEMIS:
        combined_mask[hemi] = np.logical_or.reduce([mask[hemi] for mask in masks], axis=0).astype(int)
    pickle.dump(combined_mask, open(args.path_out, mode='wb'))

    for hemi in HEMIS:
        if not args.path_out.endswith(".p"):
            raise RuntimeError("Output path must end with .p")
        path_out = args.path_out.replace(".p", f"{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(combined_mask[hemi], path_out)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paths", type=str, nargs="+", required=True)
    parser.add_argument("--path-out", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    create_masks(args)
