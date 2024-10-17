import argparse
import numpy as np
import os
from analyses.searchlight.searchlight import METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES, METRIC_MIN, METRIC_CAPTIONS, \
    METRIC_IMAGES, \
    SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR
from analyses.searchlight.searchlight_permutation_testing import load_per_subject_scores, permutation_results_dir
from preprocessing.transform_to_surface import DEFAULT_RESOLUTION
from utils import SUBJECTS, HEMIS, export_to_gifti, FS_HEMI_NAMES


def create_gifti_results_maps(args):
    print("Creating gifti results maps")
    per_subject_scores = load_per_subject_scores(args, plot_n_neighbors_correlation_graph=True)

    METRICS = [METRIC_CAPTIONS, METRIC_IMAGES, METRIC_DIFF_CAPTIONS, METRIC_DIFF_IMAGES]

    results_dir = os.path.join(permutation_results_dir(args), "acc_scores_gifti")
    os.makedirs(results_dir, exist_ok=True)

    for metric in METRICS:
        for hemi in HEMIS:
            score_hemi_avgd = np.nanmean([per_subject_scores[subj][hemi][metric] for subj in args.subjects], axis=0)
            path_out = os.path.join(results_dir, f"{metric.replace(' ', '')}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(score_hemi_avgd, path_out)

            for subj in args.subjects:
                score_hemi = per_subject_scores[subj][hemi][metric]
                path_out = os.path.join(results_dir, subj, f"{metric.replace(' ', '')}_{FS_HEMI_NAMES[hemi]}.gii")
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                export_to_gifti(score_hemi, path_out)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--model", type=str, default='imagebind')
    parser.add_argument("--features", type=str, default="avg_test_avg")

    parser.add_argument("--mod-specific-vision-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-vision-features", type=str, default="vision_test_vision")

    parser.add_argument("--mod-specific-lang-model", type=str, default='imagebind')
    parser.add_argument("--mod-specific-lang-features", type=str, default="lang_test_lang")

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--mode", type=str, default='n_neighbors_200')

    parser.add_argument("--tfce-h", type=float, default=2.0)
    parser.add_argument("--tfce-e", type=float, default=1.0)

    parser.add_argument("--metric", type=str, default=METRIC_MIN)

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(SEARCHLIGHT_PERMUTATION_TESTING_RESULTS_DIR, exist_ok=True)
    args = get_args()

    create_gifti_results_maps(args)
