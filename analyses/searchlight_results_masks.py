import argparse

import numpy as np
from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from nibabel.nifti1 import intent_codes, data_type_codes
import os
import pickle

from analyses.ridge_regression_decoding import FEATS_SELECT_DEFAULT, get_default_features, FEATURE_COMBINATION_CHOICES
from analyses.searchlight import SEARCHLIGHT_OUT_DIR
from analyses.searchlight_permutation_testing import METRIC_CODES, METRIC_MIN, calc_clusters, \
    get_edge_lengths_dicts_based_on_edges
from utils import HEMIS

FS_HEMI_NAMES = {'left': 'lh', 'right': 'rh'}


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
    p_values_path = os.path.join(
        SEARCHLIGHT_OUT_DIR, "train", args.model, args.features, args.resolution, args.mode,
        f"p_values_metric_{METRIC_CODES[args.metric]}_h_{args.tfce_h}_e_{args.tfce_e}_smoothed_{args.smoothing_iterations}.p"
    )
    p_values = pickle.load(open(p_values_path, "rb"))

    # transform to plottable magnitudes:
    p_values['left'][~np.isnan(p_values['left'])] = - np.log10(p_values['left'][~np.isnan(p_values['left'])])
    p_values['right'][~np.isnan(p_values['right'])] = - np.log10(p_values['right'][~np.isnan(p_values['right'])])

    masks_path = os.path.join(os.path.dirname(p_values_path), "masks")
    os.makedirs(masks_path, exist_ok=True)

    gifti_masks_path = os.path.join(os.path.dirname(p_values_path), "masks_gifti")
    os.makedirs(gifti_masks_path, exist_ok=True)

    for hemi in HEMIS:
        path_out = os.path.join(gifti_masks_path, f"{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(p_values[hemi], path_out)
    path_out = os.path.join(masks_path, f"p_values.p")
    pickle.dump(p_values, open(path_out, mode='wb'))

    edge_lengths = get_edge_lengths_dicts_based_on_edges(args.resolution)
    for hemi in HEMIS:
        results = calc_clusters(p_values[hemi], threshold=0, edge_lengths=edge_lengths[hemi], return_clusters=True)
        clusters = results['clusters']
        clusters.sort(key=len, reverse=True)
        for i, cluster in enumerate(clusters[:10]):
            cluster_map = np.zeros_like(p_values[hemi])
            cluster_map[list(cluster)] = 1

            path_out = os.path.join(gifti_masks_path, f"{FS_HEMI_NAMES[hemi]}_cluster_{i}.gii")
            export_to_gifti(cluster_map, path_out)

            cluster_mask = {h: np.zeros_like(cluster_map) for h in HEMIS}
            cluster_mask[hemi] = cluster_map
            path_out = os.path.join(masks_path, f"p_values_{hemi}_cluster_{i}.p")
            pickle.dump(cluster_mask, open(path_out, mode='wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='vilt')
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default='fsaverage7')
    parser.add_argument("--mode", type=str, default='n_neighbors_200')

    parser.add_argument("--smoothing-iterations", type=int, default=0)

    parser.add_argument("--tfce", action="store_true")
    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--metric", type=str, default=METRIC_MIN)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    args.features = get_default_features(args.model) if args.features == FEATS_SELECT_DEFAULT else args.features

    create_masks(args)
