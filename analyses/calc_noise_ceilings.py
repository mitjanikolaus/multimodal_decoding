import argparse
import itertools

import numpy as np
import os
import pickle

from analyses.ridge_regression_decoding import get_fmri_data, create_dissimilarity_matrix, GLM_OUT_DIR, \
    rsa_from_matrices

from utils import SUBJECTS


RSA_NOISE_CEILING_DIR = os.path.join(GLM_OUT_DIR, "noise_ceilings")


class Normalize:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.std = self.std + eps  # Avoid division by 0

    def __call__(self, x):
        return ((x - self.mean) / self.std).astype(np.float32).squeeze()


def load_mean_std(subject, mode="train"):
    mean_std_dir = os.path.join(GLM_OUT_DIR, subject)
    bold_std_mean_name = f'bold_multimodal_mean_std_{mode}.p'
    bold_std_mean_path = os.path.join(mean_std_dir, bold_std_mean_name)
    bold_mean_std = pickle.load(open(bold_std_mean_path, 'rb'))
    return Normalize(bold_mean_std['mean'], bold_mean_std['std'])


def run(args):
    matrices = dict()
    for subj in SUBJECTS:
        fmri_test_betas, _, _, _ = get_fmri_data(subj, "test", load_mean_std(subj))
        matrices[subj] = create_dissimilarity_matrix(fmri_test_betas, args.matrix_metric)

    rsa_scores = dict()
    for subj1, subj2 in itertools.combinations(SUBJECTS, 2):
        rsa = rsa_from_matrices(matrices[subj1], matrices[subj2], args.metric)
        rsa_scores[f"{subj1}_{subj2}"] = rsa

    values = list(rsa_scores.values())
    print(f"Mean RSA: {np.mean(values):.2f} Std: {np.std(values):.2f}")

    results_file = os.path.join(RSA_NOISE_CEILING_DIR, f"{args.metric}_{args.matrix_metric}.p")
    os.makedirs(RSA_NOISE_CEILING_DIR, exist_ok=True)
    pickle.dump(rsa_scores, open(results_file, "wb"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default='spearmanr')
    parser.add_argument("--matrix-metric", type=str, default='spearmanr')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(GLM_OUT_DIR, exist_ok=True)

    run(args)
