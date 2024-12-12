import argparse
import pickle

import numpy as np
import os

from tqdm import tqdm

from analyses.ridge_regression_decoding import get_run_str, FEATS_SELECT_DEFAULT, \
    get_default_features, get_default_vision_features, get_default_lang_features, FEATURE_COMBINATION_CHOICES, \
    VISION_FEAT_COMBINATION_CHOICES, LANG_FEAT_COMBINATION_CHOICES, MOD_SPECIFIC_IMAGES, MOD_SPECIFIC_CAPTIONS, \
    MODE_AGNOSTIC, TRAIN_MODE_CHOICES
from analyses.ridge_regression_encoding import ENCODER_OUT_DIR
from utils import SUBJECTS, HEMIS, export_to_gifti, FS_HEMI_NAMES, DEFAULT_RESOLUTION, DATA_DIR, \
    RESULTS_FILE, CORR_ALL, CORR_CAPTIONS, CORR_IMAGES, CORR_CROSS_CAPTIONS_TO_IMAGES, CORR_CROSS_IMAGES_TO_CAPTIONS

ENCODING_RESULTS_DIR = os.path.join(DATA_DIR, "encoding_results")
METRICS = [CORR_ALL, CORR_CAPTIONS, CORR_IMAGES, CORR_CROSS_CAPTIONS_TO_IMAGES, CORR_CROSS_IMAGES_TO_CAPTIONS]

def load_corr_scores(args, training_mode):
    per_subj_results = {}
    for subject in tqdm(args.subjects):
        per_subj_results[subject] = {}
        for hemi in HEMIS:
            per_subj_results[subject][hemi] = {}
            results_dir = os.path.join(ENCODER_OUT_DIR, training_mode, subject)
            run_str = get_run_str(args.model, args.features, args.test_features, args.vision_features,
                                  args.lang_features, mask=None, surface=True, resolution=DEFAULT_RESOLUTION,
                                  hemi=hemi)
            results_file_path = os.path.join(results_dir, run_str, RESULTS_FILE)
            results = pickle.load(open(results_file_path, "rb"))
            for metric in METRICS:
                per_subj_results[subject][hemi][metric] = results[metric]

    return per_subj_results


def calc_averaged_scores(per_subj_scores):
    scores_averaged = {}
    for hemi in HEMIS:
        scores_averaged[hemi] = {}
        for metric in METRICS:
            score_hemi_avgd = np.nanmean([per_subj_scores[subj][hemi][metric] for subj in args.subjects],
                                         axis=0)
            scores_averaged[hemi][metric] = score_hemi_avgd
    return scores_averaged


def create_gifti_results_maps(args):
    results_dir = os.path.join(ENCODING_RESULTS_DIR, "corr_results_maps")
    os.makedirs(results_dir, exist_ok=True)

    subject_scores_mod_agnostic = load_corr_scores(args, MODE_AGNOSTIC)
    averaged_scores_mod_agnostic = calc_averaged_scores(subject_scores_mod_agnostic)

    subject_scores_mod_specific_lang = load_corr_scores(args, MOD_SPECIFIC_CAPTIONS)
    averaged_scores_mod_specific_lang = calc_averaged_scores(subject_scores_mod_specific_lang)

    subject_scores_mod_specific_vision = load_corr_scores(args, MOD_SPECIFIC_IMAGES)
    averaged_scores_mod_specific_vision = calc_averaged_scores(subject_scores_mod_specific_vision)

    print("Creating gifti results maps")

    for metric in METRICS:
        for hemi in HEMIS:
            print(f"{metric} ({hemi} hemi) mean over subjects: {np.nanmean(averaged_scores_mod_agnostic[hemi][metric]):.3f}")
            path_out = os.path.join(results_dir, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(averaged_scores_mod_agnostic[hemi][metric], path_out)

            for subj in args.subjects:
                score_hemi = subject_scores_mod_agnostic[subj][hemi][metric]
                path_out = os.path.join(results_dir, subj, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                print(f"{os.path.basename(path_out)} max: {np.max(score_hemi)}")
                export_to_gifti(score_hemi, path_out)

                score_hemi_specific_lang = subject_scores_mod_specific_lang[subj][hemi][metric]
                path_out = os.path.join(results_dir, subj, f"mod_specific_lang_{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                print(f"{os.path.basename(path_out)} max: {np.max(score_hemi_specific_lang)}")
                export_to_gifti(score_hemi_specific_lang, path_out)

                score_hemi_specific_vision = subject_scores_mod_specific_vision[subj][hemi][metric]
                path_out = os.path.join(results_dir, subj, f"mod_specific_vision_{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                print(f"{os.path.basename(path_out)} max: {np.max(score_hemi_specific_vision)}")
                export_to_gifti(score_hemi_specific_vision, path_out)

                cross_lang_to_vision = subject_scores_mod_specific_lang[subj][hemi][CORR_CROSS_CAPTIONS_TO_IMAGES]
                path_out = os.path.join(results_dir, subj, f"cross_lang_to_vision_{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                print(f"{os.path.basename(path_out)} max: {np.max(cross_lang_to_vision)}")
                export_to_gifti(cross_lang_to_vision, path_out)

                cross_vision_to_lang = subject_scores_mod_specific_vision[subj][hemi][CORR_CROSS_IMAGES_TO_CAPTIONS]
                path_out = os.path.join(results_dir, subj, f"cross_vision_to_lang_{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                print(f"{os.path.basename(path_out)} max: {np.max(cross_vision_to_lang)}")
                export_to_gifti(cross_vision_to_lang, path_out)


    for hemi in HEMIS:
        diff_corr_captions_mod_agnositic_mod_specific = averaged_scores_mod_agnostic[hemi][CORR_CAPTIONS] - averaged_scores_mod_specific_lang[hemi][CORR_CAPTIONS]
        diff_corr_images_mod_agnositic_mod_specific = averaged_scores_mod_agnostic[hemi][CORR_IMAGES] - averaged_scores_mod_specific_vision[hemi][CORR_IMAGES]
        diff_mod_agnositic_mod_specific = np.min([diff_corr_captions_mod_agnositic_mod_specific, diff_corr_images_mod_agnositic_mod_specific], axis=0)

        path_out = os.path.join(results_dir, f"diff_corr_captions_mod_agnositic_mod_specific_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(diff_corr_captions_mod_agnositic_mod_specific, path_out)

        path_out = os.path.join(results_dir, f"diff_corr_images_mod_agnositic_mod_specific_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(diff_corr_images_mod_agnositic_mod_specific, path_out)

        path_out = os.path.join(results_dir, f"diff_mod_agnositic_mod_specific_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(diff_mod_agnositic_mod_specific, path_out)

        cross_encoding = np.min([averaged_scores_mod_specific_lang[hemi][CORR_CROSS_CAPTIONS_TO_IMAGES], averaged_scores_mod_specific_vision[hemi][CORR_CROSS_IMAGES_TO_CAPTIONS]], axis=0)
        path_out = os.path.join(results_dir, f"cross_encoding_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(cross_encoding, path_out)

        cross_encoding_or_diff = np.max([cross_encoding, diff_mod_agnositic_mod_specific], axis=0)
        path_out = os.path.join(results_dir, f"cross_encoding_or_diff_mod_agnostic_mod_specific_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(cross_encoding_or_diff, path_out)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--model", type=str, default='imagebind')
    parser.add_argument("--features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--test-features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--vision-features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--mode", type=str, default='n_neighbors_750')

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(ENCODING_RESULTS_DIR, exist_ok=True)
    args = get_args()

    model_name = args.model
    if args.features == FEATS_SELECT_DEFAULT:
        args.features = get_default_features(model_name)
    if args.vision_features == FEATS_SELECT_DEFAULT:
        args.vision_features = get_default_vision_features(model_name)
    if args.lang_features == FEATS_SELECT_DEFAULT:
        args.lang_features = get_default_lang_features(model_name)
    if args.test_features == FEATS_SELECT_DEFAULT:
        args.test_features = get_default_features(model_name)

    create_gifti_results_maps(args)
