import argparse
import pickle

import numpy as np
import os

from tqdm import tqdm

from analyses.encoding.encoding_permutation_testing import CORR_IMAGES_MOD_SPECIFIC_IMAGES, \
    CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS
from analyses.encoding.ridge_regression_encoding import ENCODING_RESULTS_DIR, get_results_file_path
from data import SELECT_DEFAULT, VISION_FEAT_COMBINATION_CHOICES, \
    LANG_FEAT_COMBINATION_CHOICES, LatentFeatsConfig, MODALITY_AGNOSTIC, MODALITY_SPECIFIC_CAPTIONS, \
    MODALITY_SPECIFIC_IMAGES, FEATURE_COMBINATION_CHOICES
from eval import CORR_ALL, CORR_CAPTIONS, CORR_IMAGES
from utils import SUBJECTS, HEMIS, export_to_gifti, FS_HEMI_NAMES, DEFAULT_RESOLUTION, DEFAULT_MODEL

METRICS = [CORR_ALL, CORR_CAPTIONS, CORR_IMAGES]


def load_corr_scores(args, training_mode, feats_config):
    per_subj_results = {}
    for subject in tqdm(args.subjects):
        per_subj_results[subject] = {}
        for hemi in HEMIS:
            per_subj_results[subject][hemi] = {}
            results_file_path = get_results_file_path(subject, training_mode, feats_config,
                                                      args.resolution, hemi)

            results = pickle.load(open(results_file_path, "rb"))
            nan_locations = results['nan_locations']
            for metric in METRICS:
                per_subj_results[subject][hemi][metric] = np.repeat(np.nan, nan_locations.shape)
                per_subj_results[subject][hemi][metric][~nan_locations] = results[metric]

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
    feats_config_agnostic = LatentFeatsConfig(
        args.model, args.features, args.test_features, args.vision_features, args.lang_features
    )
    subject_scores_mod_agnostic = load_corr_scores(args, MODALITY_AGNOSTIC, feats_config_agnostic)
    averaged_scores_mod_agnostic = calc_averaged_scores(subject_scores_mod_agnostic)

    feats_config_specific_lang = LatentFeatsConfig(
        args.mod_specific_captions_model, args.mod_specific_captions_features, args.mod_specific_captions_test_features,
        args.vision_features, args.lang_features
    )
    subject_scores_mod_specific_captions = load_corr_scores(args, MODALITY_SPECIFIC_CAPTIONS,
                                                            feats_config_specific_lang)
    averaged_scores_mod_specific_captions = calc_averaged_scores(subject_scores_mod_specific_captions)

    feats_config_specific_vision = LatentFeatsConfig(
        args.mod_specific_images_model, args.mod_specific_images_features, args.mod_specific_images_test_features,
        args.vision_features, args.lang_features
    )
    subject_scores_mod_specific_images = load_corr_scores(args, MODALITY_SPECIFIC_IMAGES, feats_config_specific_vision)
    averaged_scores_mod_specific_images = calc_averaged_scores(subject_scores_mod_specific_images)

    print("Creating gifti results maps")
    for metric in METRICS:
        for hemi in HEMIS:
            print(
                f"{metric} ({hemi} hemi) mean over subjects: {np.nanmean(averaged_scores_mod_agnostic[hemi][metric]):.3f}")
            path_out = os.path.join(results_dir, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
            export_to_gifti(averaged_scores_mod_agnostic[hemi][metric], path_out)

            for subj in args.subjects:
                score_hemi = subject_scores_mod_agnostic[subj][hemi][metric]
                path_out = os.path.join(results_dir, subj, f"{metric}_{FS_HEMI_NAMES[hemi]}.gii")
                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                export_to_gifti(score_hemi, path_out)

    for subj in args.subjects:
        print(f"\n{subj}")
        for hemi in HEMIS:
            score_hemi_specific_lang = subject_scores_mod_specific_captions[subj][hemi][CORR_CAPTIONS]
            path_out = os.path.join(results_dir, subj, f"mod_specific_captions_captions_{FS_HEMI_NAMES[hemi]}.gii")
            print(f"{os.path.basename(path_out)} max: {np.nanmax(score_hemi_specific_lang):.2f}")
            export_to_gifti(score_hemi_specific_lang, path_out)

            score_hemi_specific_vision = subject_scores_mod_specific_images[subj][hemi][CORR_IMAGES]
            path_out = os.path.join(results_dir, subj, f"mod_specific_images_images_{FS_HEMI_NAMES[hemi]}.gii")
            print(f"{os.path.basename(path_out)} max: {np.nanmax(score_hemi_specific_vision):.2f}")
            export_to_gifti(score_hemi_specific_vision, path_out)

            cross_lang_to_vision = subject_scores_mod_specific_captions[subj][hemi][CORR_IMAGES]
            path_out = os.path.join(results_dir, subj, f"cross_lang_to_vision_{FS_HEMI_NAMES[hemi]}.gii")
            print(f"{os.path.basename(path_out)} max: {np.nanmax(cross_lang_to_vision):.2f}")
            export_to_gifti(cross_lang_to_vision, path_out)

            cross_vision_to_lang = subject_scores_mod_specific_images[subj][hemi][CORR_CAPTIONS]
            path_out = os.path.join(results_dir, subj, f"cross_vision_to_lang_{FS_HEMI_NAMES[hemi]}.gii")
            print(f"{os.path.basename(path_out)} max: {np.nanmax(cross_vision_to_lang):.2f}")
            export_to_gifti(cross_vision_to_lang, path_out)

    for hemi in HEMIS:
        diff_corr_captions_mod_agnositic_mod_specific = averaged_scores_mod_agnostic[hemi][CORR_CAPTIONS] - \
                                                        averaged_scores_mod_specific_captions[hemi][CORR_CAPTIONS]
        diff_corr_images_mod_agnositic_mod_specific = averaged_scores_mod_agnostic[hemi][CORR_IMAGES] - \
                                                      averaged_scores_mod_specific_images[hemi][CORR_IMAGES]
        diff_mod_agnositic_mod_specific = np.nanmin(
            [diff_corr_captions_mod_agnositic_mod_specific, diff_corr_images_mod_agnositic_mod_specific], axis=0)

        path_out = os.path.join(results_dir, f"{CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(averaged_scores_mod_specific_captions[hemi][CORR_CAPTIONS], path_out)

        path_out = os.path.join(results_dir, f"{CORR_IMAGES_MOD_SPECIFIC_IMAGES}_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(averaged_scores_mod_specific_images[hemi][CORR_IMAGES], path_out)

        path_out = os.path.join(results_dir, f"diff_corr_captions_mod_agnositic_mod_specific_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(diff_corr_captions_mod_agnositic_mod_specific, path_out)

        path_out = os.path.join(results_dir, f"diff_corr_images_mod_agnositic_mod_specific_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(diff_corr_images_mod_agnositic_mod_specific, path_out)

        path_out = os.path.join(results_dir, f"diff_mod_agnositic_mod_specific_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(diff_mod_agnositic_mod_specific, path_out)

        cross_encoding = np.nanmin([averaged_scores_mod_specific_captions[hemi][CORR_IMAGES],
                                    averaged_scores_mod_specific_images[hemi][CORR_CAPTIONS]], axis=0)
        path_out = os.path.join(results_dir, f"cross_encoding_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(cross_encoding, path_out)

        cross_encoding_or_diff = np.nanmax([cross_encoding, diff_mod_agnositic_mod_specific], axis=0)
        path_out = os.path.join(results_dir,
                                f"cross_encoding_or_diff_mod_agnostic_mod_specific_{FS_HEMI_NAMES[hemi]}.gii")
        export_to_gifti(cross_encoding_or_diff, path_out)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs="+", default=SUBJECTS)

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--mod-specific-images-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-images-features", type=str, default=SELECT_DEFAULT)
    parser.add_argument("--mod-specific-images-test-features", type=str, default=SELECT_DEFAULT)

    parser.add_argument("--mod-specific-captions-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--mod-specific-captions-features", type=str, default=SELECT_DEFAULT)
    parser.add_argument("--mod-specific-captions-test-features", type=str, default=SELECT_DEFAULT)

    parser.add_argument("--vision-features", type=str, default=SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)
    parser.add_argument("--mode", type=str, default='n_neighbors_750')

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs(ENCODING_RESULTS_DIR, exist_ok=True)
    args = get_args()

    model_name = args.model
    create_gifti_results_maps(args)
