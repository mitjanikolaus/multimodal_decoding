import argparse
import itertools

import numpy as np
import os
import pickle

from analyses.ridge_regression_decoding import get_fmri_data, DECODER_OUT_DIR, calc_rsa, calc_rsa_images, calc_rsa_captions, \
    create_dissimilarity_matrix, rsa_from_matrices

from utils import SUBJECTS


RSA_NOISE_CEILING_DIR = os.path.join(DECODER_OUT_DIR, "noise_ceilings")


class Normalize:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.std = self.std + eps  # Avoid division by 0

    def __call__(self, x):
        return ((x - self.mean) / self.std).astype(np.float32).squeeze()


def load_mean_std(subject, mode="train"):
    mean_std_dir = os.path.join(DECODER_OUT_DIR, subject)
    bold_std_mean_name = f'bold_multimodal_mean_std_{mode}.p'
    bold_std_mean_path = os.path.join(mean_std_dir, bold_std_mean_name)
    bold_mean_std = pickle.load(open(bold_std_mean_path, 'rb'))
    return Normalize(bold_mean_std['mean'], bold_mean_std['std'])


def run(args):
    all_betas = dict()
    all_stim_types = dict()
    all_stim_ids = dict()

    for subj in SUBJECTS:
        fmri_test_betas, stim_ids, stim_types, _ = get_fmri_data(subj, "test", load_mean_std(subj))
        all_betas[subj] = fmri_test_betas
        all_stim_types[subj] = stim_types
        all_stim_ids[subj] = stim_ids

    rsa_scores = dict()
    rsa_images_scores = dict()
    rsa_captions_scores = dict()

    for subj1, subj2 in itertools.combinations(SUBJECTS, 2):
        assert np.all(all_stim_ids[subj1] == all_stim_ids[subj2])

        rsa_scores[f"{subj1}_{subj2}"] = calc_rsa(all_betas[subj1], all_betas[subj2], args.metric, args.matrix_metric)

        rsa_images_scores[f"{subj1}_{subj2}"] = calc_rsa_images(all_betas[subj1], all_betas[subj2], all_stim_types[subj1], args.metric, args.matrix_metric)
        rsa_captions_scores[f"{subj1}_{subj2}"] = calc_rsa_captions(all_betas[subj1], all_betas[subj2], all_stim_types[subj1], args.metric, args.matrix_metric)

    values = list(rsa_scores.values())
    print(f"Between-subject RSA: {np.mean(values):.2f} Std: {np.std(values):.2f}")

    values = list(rsa_images_scores.values())
    print(f"Between-subject RSA (images): {np.mean(values):.2f} Std: {np.std(values):.2f}")

    values = list(rsa_captions_scores.values())
    print(f"Between-subject RSA (captions): {np.mean(values):.2f} Std: {np.std(values):.2f}")

    all_rsa_ceilings = {"rsa": rsa_scores, "rsa_images": rsa_images_scores, "rsa_captions": rsa_captions_scores}
    results_file = os.path.join(RSA_NOISE_CEILING_DIR, f"{args.metric}_{args.matrix_metric}.p")
    os.makedirs(RSA_NOISE_CEILING_DIR, exist_ok=True)
    pickle.dump(all_rsa_ceilings, open(results_file, "wb"))

    rsa_img_caps = []
    for subj in SUBJECTS:
        betas_captions = all_betas[subj][all_stim_types[subj] == 'caption']
        betas_images = all_betas[subj][all_stim_types[subj] == 'image']
        matrix_images = create_dissimilarity_matrix(betas_images, matrix_metric=args.matrix_metric)
        matrix_captions = create_dissimilarity_matrix(betas_captions, matrix_metric=args.matrix_metric)
        rsa_img_caps.append(rsa_from_matrices(matrix_images, matrix_captions, args.metric))

    print(f"RSA imgs-caps: {np.mean(rsa_img_caps):.2f} Std: {np.std(rsa_img_caps):.2f}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default='spearmanr')
    parser.add_argument("--matrix-metric", type=str, default='spearmanr')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(DECODER_OUT_DIR, exist_ok=True)

    run(args)
