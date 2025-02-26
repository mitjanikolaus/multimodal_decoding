import argparse

import numpy as np
import os

from analyses.encoding.encoding_permutation_testing import CORR_IMAGES_MOD_SPECIFIC_IMAGES, \
    CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS, add_encoding_permutation_args, T_VAL_METRICS, permutation_results_dir, \
    load_per_subject_scores
from analyses.encoding.ridge_regression_encoding import ENCODING_RESULTS_DIR
from eval import CORR_IMAGES_MOD_SPECIFIC_CAPTIONS, METRIC_CROSS_ENCODING, \
    CORR_CAPTIONS_MOD_SPECIFIC_IMAGES, CORR_IMAGES_MOD_AGNOSTIC, CORR_CAPTIONS_MOD_AGNOSTIC
from utils import HEMIS, export_to_gifti, FS_HEMI_NAMES, METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC, \
    METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC, METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC

METRICS = T_VAL_METRICS


def create_gifti_results_maps(args):
    results_dir = os.path.join(permutation_results_dir(args), "acc_results_maps")
    os.makedirs(results_dir, exist_ok=True)

    print("Creating gifti results maps")
    subject_scores, nan_locations, n_neighbors = load_per_subject_scores(
        args,
    )

    subject_scores_avgd = {hemi: dict() for hemi in HEMIS}
    for metric in METRICS:
        for hemi in HEMIS:
            for subj in args.subjects:
                if metric in subject_scores[subj][hemi]:
                    score_hemi = subject_scores[subj][hemi][metric]
                    path_out = os.path.join(results_dir, subj, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                    os.makedirs(os.path.dirname(path_out), exist_ok=True)
                    export_to_gifti(score_hemi, path_out)
                else:
                    print(f"missing metric: {subj} {metric} {hemi}")

            if metric in subject_scores[args.subjects[-1]][hemi]:
                subject_scores_avgd[hemi][metric] = np.mean(  # TODO at least 3 datapoints?
                    [subject_scores[subj][hemi][metric] for subj in args.subjects], axis=0)
                print(f"{metric} ({hemi} hemi) mean over subjects: {np.nanmean(subject_scores_avgd[hemi][metric])}")
                path_out = os.path.join(results_dir, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                export_to_gifti(subject_scores_avgd[hemi][metric], path_out)
            else:
                print(f"missing metric: {args.subjects[-1]} {metric} {hemi}")

    for hemi in HEMIS:
        for subj in args.subjects:
            subject_scores[subj][hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC] = np.nanmin(
                (
                    subject_scores[subj][hemi][METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC],
                    subject_scores[subj][hemi][METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC],
                    subject_scores[subj][hemi][CORR_IMAGES_MOD_AGNOSTIC],
                    subject_scores[subj][hemi][CORR_CAPTIONS_MOD_AGNOSTIC]),
                axis=0)
            path_out = os.path.join(results_dir, subj,
                                    f"{METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(subject_scores[subj][hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC], path_out)

            subject_scores[subj][hemi][METRIC_CROSS_ENCODING] = np.nanmin(
                (subject_scores[subj][hemi][CORR_IMAGES_MOD_SPECIFIC_CAPTIONS],
                 subject_scores[subj][hemi][CORR_CAPTIONS_MOD_SPECIFIC_IMAGES],
                 subject_scores[subj][hemi][CORR_IMAGES_MOD_SPECIFIC_IMAGES],
                 subject_scores[subj][hemi][CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS]),
                axis=0
            )
            path_out = os.path.join(results_dir, subj, f"{METRIC_CROSS_ENCODING}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(subject_scores[subj][hemi][METRIC_CROSS_ENCODING], path_out)

        subject_scores_avgd[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC] = np.nanmin(
            (
                subject_scores_avgd[hemi][METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC],
                subject_scores_avgd[hemi][METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC],
                subject_scores_avgd[hemi][CORR_IMAGES_MOD_AGNOSTIC],
                subject_scores_avgd[hemi][CORR_CAPTIONS_MOD_AGNOSTIC]),
            axis=0)
        path_out = os.path.join(results_dir, f"{METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(subject_scores_avgd[hemi][METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC], path_out)

        subject_scores_avgd[hemi][METRIC_CROSS_ENCODING] = np.nanmin(
            (subject_scores_avgd[hemi][CORR_IMAGES_MOD_SPECIFIC_CAPTIONS],
             subject_scores_avgd[hemi][CORR_CAPTIONS_MOD_SPECIFIC_IMAGES],
             subject_scores_avgd[hemi][CORR_IMAGES_MOD_SPECIFIC_IMAGES],
             subject_scores_avgd[hemi][CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS]),
            axis=0
        )
        path_out = os.path.join(results_dir, f"{METRIC_CROSS_ENCODING}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(subject_scores_avgd[hemi][METRIC_CROSS_ENCODING], path_out)


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_encoding_permutation_args(parser)

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(ENCODING_RESULTS_DIR, exist_ok=True)
    args = get_args()

    model_name = args.model
    create_gifti_results_maps(args)
