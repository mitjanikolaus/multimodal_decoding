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

    tfce_values_flat = np.concatenate((tfce_values[HEMIS[0][metric]], tfce_values[HEMIS[1][metric]]))

    if args.n_vertices is not None:
        fname = f"{metric}_{args.n_vertices}_vertices.p"

        mask = np.zeros(shape=tfce_values_flat.shape, dtype=np.uint8)
        print(f"Creating {metric} mask with {args.n_vertices} vertices")
        indices = np.argsort(tfce_values_flat)[-args.n_vertices:]
        mask[indices] = 1
    else:
        thresh = args.tfce_value_threshold
        fname = f"{metric}_threshold_{round(thresh)}.p"

        mask = copy.deepcopy(tfce_values_flat)
        print(f"Creating {metric} mask with tfce value threshold: {args.tfce_value_threshold}")
        print(f'mask size for threshold {thresh:.2f}: {np.mean(tfce_values >= thresh):.2f}')
        mask[tfce_values_flat >= thresh] = 1
        mask[tfce_values_flat < thresh] = 0
        mask[np.isnan(tfce_values_flat)] = 0
        mask = mask.astype(np.uint8)

    mask_path = os.path.join(results_dir, 'masks')
    os.makedirs(mask_path, exist_ok=True)
    pickle.dump(mask, open(os.path.join(mask_path, fname), "wb"))
    print(f'saved {os.path.join(mask_path, fname)}')


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_searchlight_permutation_args(parser)

    # parser.add_argument("--p-value-threshold", type=float, default=DEFAULT_P_VAL_THRESHOLD)
    parser.add_argument("--n-vertices", type=int, default=DEFAULT_NUM_VERTICES)
    parser.add_argument("--tfce-value-threshold", type=float, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    for metric in args.metrics:
        args.metric = metric
        create_masks(args)
