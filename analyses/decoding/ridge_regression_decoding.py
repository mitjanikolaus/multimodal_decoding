import argparse
import itertools
import time

import numpy as np
import os
import pickle

from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from data import LatentFeatsConfig, SELECT_DEFAULT, FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, \
    LANG_FEAT_COMBINATION_CHOICES, apply_mask, standardize_fmri_betas, get_latent_features, \
    standardize_latents, MODALITY_AGNOSTIC, TRAINING_MODES, SPLIT_TRAIN, get_fmri_data, \
    ALL_SPLITS, TEST_SPLITS, get_latents_for_splits, SPLIT_IMAGERY_WEAK
from eval import pairwise_accuracy, calc_all_pairwise_accuracy_scores
from utils import FMRI_BETAS_DIR, SUBJECTS, RESULTS_FILE, DEFAULT_MODEL, DEFAULT_RESOLUTION, \
    DECODER_ADDITIONAL_TEST_OUT_DIR, PREDICTIONS_FILE, HEMIS

NUM_CV_SPLITS = 5
DEFAULT_ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


def get_run_str(betas_dir, feats_config, mask=None, surface=False, resolution=DEFAULT_RESOLUTION,
                training_splits=[SPLIT_TRAIN], imagery_samples_weight=None):
    run_str = f"{feats_config.model}_{feats_config.combined_feats}"
    run_str += f"_{feats_config.vision_features}"
    run_str += f"_{feats_config.lang_features}"
    if betas_dir.endswith(os.sep):
        betas_dir = betas_dir[:-1]
    run_str += f"_{betas_dir.split(os.sep)[-1]}"

    if mask is not None:
        if mask.startswith("functional_") or mask.startswith("anatomical_"):
            run_str += f"_mask_{mask}"
        else:
            mask_name = os.path.basename(mask).replace(".p", "")
            run_str += f"_mask_{mask_name}"

    if surface:
        run_str += f"_surface_{resolution}"

    if (len(training_splits) > 1) or (training_splits[0] != SPLIT_TRAIN):
        run_str += f"_train_splits_{'_'.join(training_splits)}"

    if imagery_samples_weight is not None:
        run_str += f"_imagery_samples_weight_{imagery_samples_weight}"

    return run_str


def get_fmri_data_for_splits(subject, splits, training_mode, betas_dir, surface=False, hemis=HEMIS):
    fmri_betas, stim_ids, stim_types = dict(), dict(), dict()
    for split in splits:
        mode = training_mode if split == SPLIT_TRAIN else MODALITY_AGNOSTIC
        fmri_betas[split], stim_ids[split], stim_types[split] = get_fmri_data(
            betas_dir,
            subject,
            split,
            mode,
            surface=surface,
            hemis=hemis,
        )

    return fmri_betas, stim_ids, stim_types


def run(args):
    for training_mode in args.training_modes:
        for subject in args.subjects:
            fmri_betas_full, stim_ids, stim_types = get_fmri_data_for_splits(
                subject, ALL_SPLITS, training_mode, args.betas_dir, args.surface,
            )
            for mask in args.masks:
                mask = None if mask in ["none", "None"] else mask
                fmri_betas = apply_mask(mask, fmri_betas_full, args)
                fmri_betas = standardize_fmri_betas(fmri_betas)
                for split in fmri_betas.keys():
                    print(f"{split} fMRI betas shape: {fmri_betas[split].shape}")

                for model in args.models:
                    feats_config = LatentFeatsConfig(
                        model, args.features, args.test_features, args.vision_features, args.lang_features
                    )

                    print(f"TRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                          f"MODEL: {model} | FEATURES: {feats_config.features} {feats_config.vision_features} "
                          f"{feats_config.lang_features} | TEST FEATURES: {feats_config.test_features}")
                    if mask is not None:
                        print("Mask: ", os.path.basename(mask))

                    run_str = get_run_str(args.betas_dir, feats_config, mask, args.surface, args.resolution,
                                          args.training_splits)
                    results_file_path = os.path.join(
                        DECODER_ADDITIONAL_TEST_OUT_DIR, training_mode, subject, run_str, RESULTS_FILE
                    )
                    if os.path.isfile(results_file_path) and not args.overwrite:
                        print(f"Skipping decoder training as results are already present at {results_file_path}")
                        continue

                    latents = get_latents_for_splits(subject, feats_config, ALL_SPLITS, training_mode)
                    latents = standardize_latents(latents)
                    print(f"train latents shape: {latents[SPLIT_TRAIN].shape}")

                    pairwise_acc_scorer = make_scorer(pairwise_accuracy, greater_is_better=True)

                    clf = GridSearchCV(
                        estimator=Ridge(fit_intercept=False),
                        param_grid=dict(alpha=args.l2_regularization_alphas),
                        scoring=pairwise_acc_scorer, cv=NUM_CV_SPLITS, n_jobs=args.n_jobs,
                        pre_dispatch=args.n_pre_dispatch, refit=True, verbose=3,
                    )

                    start = time.time()
                    fmri_betas_train = np.concatenate([fmri_betas[split] for split in args.training_splits])

                    latents_train = np.concatenate([latents[split] for split in args.training_splits])
                    print(f"Training set size: {len(fmri_betas_train)} (splits: {args.training_splits})")

                    fit_params = dict()
                    if args.imagery_samples_weight is not None:
                        fit_params['sample_weight'] = list(itertools.chain(*[
                            [args.imagery_samples_weight] * len(fmri_betas[split]) if split == SPLIT_IMAGERY_WEAK else [1] * len(
                                split) for split in args.training_splits]))
                        print('applying sample weights: ', fit_params['sample_weight'])
                    clf.fit(fmri_betas_train, latents_train, **fit_params)
                    end = time.time()
                    print(f"Elapsed time: {int(end - start)}s")

                    best_alpha = clf.best_params_["alpha"]

                    best_model = clf.best_estimator_

                    predicted_latents = {split: best_model.predict(fmri_betas[split]) for split in TEST_SPLITS}

                    scores_df = calc_all_pairwise_accuracy_scores(latents, predicted_latents,
                                                                  standardize_predictions_conds=[True, False])
                    scores_df["model"] = model
                    scores_df["subject"] = subject
                    scores_df["features"] = feats_config.features
                    scores_df["test_features"] = feats_config.test_features
                    scores_df["vision_features"] = feats_config.vision_features
                    scores_df["lang_features"] = feats_config.lang_features
                    scores_df["training_mode"] = training_mode
                    scores_df["training_splits"] = '_'.join(args.training_splits)
                    scores_df["mask"] = mask
                    scores_df["num_voxels"] = fmri_betas[SPLIT_TRAIN].shape[1]
                    scores_df["surface"] = args.surface
                    scores_df["resolution"] = args.resolution
                    scores_df["imagery_samples_weight"] = args.imagery_samples_weight

                    print(
                        f"Best alpha: {best_alpha}"
                    )

                    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                    scores_df.to_csv(results_file_path, index=False)

                    predictions_file_path = os.path.join(
                        DECODER_ADDITIONAL_TEST_OUT_DIR, training_mode, subject, run_str, PREDICTIONS_FILE
                    )
                    pickle.dump(predicted_latents, open(predictions_file_path, 'wb'))

                    scores_to_print = scores_df[scores_df.standardized_predictions == True]
                    scores_to_print = scores_to_print[scores_to_print.latents == 'all_candidate_latents']
                    print(scores_to_print.to_string(index=False))
                    print('\n\n')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--training-splits", type=str, nargs="+", default=[SPLIT_TRAIN])
    parser.add_argument("--imagery-samples-weight", type=int, default=None)

    parser.add_argument("--training-modes", type=str, nargs="+", default=[MODALITY_AGNOSTIC],
                        choices=TRAINING_MODES)

    parser.add_argument("--models", type=str, nargs='+', default=[DEFAULT_MODEL])
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--vision-features", type=str, default=SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+', default=DEFAULT_ALPHAS)

    parser.add_argument("--n-jobs", type=int, default=10)
    parser.add_argument("--n-pre-dispatch", type=int, default=10)

    parser.add_argument("--surface", action='store_true', default=False)
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument("--masks", nargs='+', type=str, default=[None])

    parser.add_argument("--overwrite", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run(args)
