import argparse
import copy
import pickle

import numpy as np
import os

from analyses.decoding.searchlight.searchlight_permutation_testing import permutation_results_dir, \
    add_searchlight_permutation_args
from utils import HEMIS

DEFAULT_NUM_VERTICES = 1000


def create_masks(args):
    results_dir = permutation_results_dir(args)
    tfce_values_path = os.path.join(results_dir, f"tfce_values_{metric}.p")
    tfce_values = pickle.load(open(tfce_values_path, "rb"))

    if args.num_vertices is not None:
        fname = f"{metric}_{args.num_vertices}_vertices.p"

        masks = {hemi: np.zeros(shape=tfce_values[hemi][metric].shape, dtype=np.uint8) for hemi in HEMIS}
        print(f"Creating {metric} mask with {args.num_vertices} vertices")
        for hemi in HEMIS:
            indices = np.argsort(tfce_values[hemi][metric])[-args.num_vertices:]
            masks[hemi][indices] = 1
    else:
        thresh = args.tfce_value_threshold
        fname = f"{metric}_threshold_{round(thresh)}.p"

        masks = {hemi: copy.deepcopy(tfce_values[hemi][metric]) for hemi in HEMIS}
        print(f"Creating {metric} mask with tfce value threshold: {args.tfce_value_threshold}")
        for hemi in HEMIS:
            print(
                f'{hemi} hemi mask size for threshold {thresh:.2f}: {np.mean(tfce_values[hemi][metric] >= thresh):.2f}')
            masks[hemi][tfce_values[hemi][metric] >= thresh] = 1
            masks[hemi][tfce_values[hemi][metric] < thresh] = 0
            masks[hemi][np.isnan(tfce_values[hemi][metric])] = 0
            masks[hemi] = masks[hemi].astype(np.uint8)

    mask_path = os.path.join(results_dir, 'masks')
    os.makedirs(mask_path, exist_ok=True)
    pickle.dump(masks, open(os.path.join(mask_path, fname), "wb"))
    print(f'saved {mask_path}')


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    # parser.add_argument("--p-value-threshold", type=float, default=DEFAULT_P_VAL_THRESHOLD)
    parser.add_argument("--num-vertices", type=int, default=DEFAULT_NUM_VERTICES)
    parser.add_argument("--tfce-value-threshold", type=float, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    for metric in args.metrics:
        args.metric = metric
        create_masks(args)
