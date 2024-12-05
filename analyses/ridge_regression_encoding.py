import argparse
import time

import numpy as np
import nibabel as nib
import torch.cuda
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV, Ridge
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import os
from glob import glob
import pickle
from tqdm import trange

from analyses.ridge_regression_decoding import get_fmri_data, TESTING_MODE, IMAGERY, apply_mask_and_clean, \
    standardize_fmri_betas, FEATS_SELECT_DEFAULT, get_default_features, get_default_vision_features, \
    get_default_lang_features, get_run_str, get_nn_latent_data, DEFAULT_N_JOBS, DEFAULT_N_PRE_DISPATCH, \
    LANG_FEAT_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, FEATURE_COMBINATION_CHOICES, TRAIN_MODE_CHOICES
from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import IMAGERY_SCENES, FMRI_BETAS_DIR, model_features_file_path, VISION_MEAN_FEAT_KEY, \
    VISION_CLS_FEAT_KEY, FUSED_CLS_FEAT_KEY, FUSED_MEAN_FEAT_KEY, LANG_MEAN_FEAT_KEY, \
    LANG_CLS_FEAT_KEY, FMRI_SURFACE_LEVEL_DIR, HEMIS, SUBJECTS, ACC_CAPTIONS, ACC_IMAGES, \
    ACC_CROSS_CAPTIONS_TO_IMAGES, ACC_CROSS_IMAGES_TO_CAPTIONS, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, \
    ACC_MODALITY_AGNOSTIC, DEFAULT_RESOLUTION


ENCODER_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/whole_brain_encoding/")


def run(args):
    if torch.cuda.is_available():
        print("Setting backend to cuda")
        backend = set_backend("torch_cuda")
    else:
        backend = set_backend("numpy")

    for training_mode in args.training_modes:
        for subject in args.subjects:
            train_fmri_betas_full, train_stim_ids, train_stim_types = get_fmri_data(
                subject,
                training_mode,
                surface=args.surface,
                resolution=args.resolution,
            )
            test_fmri_betas_full, test_stim_ids, test_stim_types = get_fmri_data(
                subject,
                TESTING_MODE,
                surface=args.surface,
                resolution=args.resolution,
            )
            imagery_fmri_betas_full, imagery_stim_ids, imagery_stim_types = get_fmri_data(
                subject,
                IMAGERY,
                surface=args.surface,
                resolution=args.resolution,
            )
            for mask in args.masks:
                mask = None if mask in ["none", "None"] else mask
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = apply_mask_and_clean(
                    mask, [train_fmri_betas_full, test_fmri_betas_full, imagery_fmri_betas_full], args
                )
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = standardize_fmri_betas(
                    train_fmri_betas, test_fmri_betas, imagery_fmri_betas, subject, training_mode, mask
                )

                for model_name in args.models:
                    model_name = model_name.lower()

                    for features in args.features:
                        if features == FEATS_SELECT_DEFAULT:
                            features = get_default_features(model_name)
                        test_features = args.test_features
                        if test_features == FEATS_SELECT_DEFAULT:
                            test_features = get_default_features(model_name)

                        for vision_features in args.vision_features:
                            if vision_features == FEATS_SELECT_DEFAULT:
                                vision_features = get_default_vision_features(model_name)
                            for lang_features in args.lang_features:
                                if lang_features == FEATS_SELECT_DEFAULT:
                                    lang_features = get_default_lang_features(model_name)

                                print(f"\nTRAIN MODE: {training_mode} | MASK: {mask} | SUBJECT: {subject} | "
                                      f"MODEL: {model_name} | FEATURES: {features} {vision_features} {lang_features} | "
                                      f"TEST FEATURES: {test_features}")
                                print(f"train fMRI betas shape: {train_fmri_betas.shape}")
                                print(f"test fMRI betas shape: {test_fmri_betas.shape}")
                                print(f"imagery fMRI betas shape: {imagery_fmri_betas.shape}")

                                results_dir = os.path.join(ENCODER_OUT_DIR, training_mode, subject)
                                run_str = get_run_str(
                                    model_name, features, test_features, vision_features, lang_features, mask,
                                    args.surface,
                                    args.resolution)
                                results_file_path = os.path.join(results_dir, run_str, "results.p")
                                if os.path.isfile(results_file_path) and not args.overwrite:
                                    print(f"Skipping encoder training as results are already present at"
                                          f" {results_file_path}")
                                    continue

                                train_latents, latent_transform = get_nn_latent_data(
                                    model_name, features,
                                    vision_features,
                                    lang_features,
                                    train_stim_ids,
                                    train_stim_types,
                                    subject,
                                    training_mode,
                                )

                                model = RidgeCV(alphas=args.l2_regularization_alphas)

                                train_fmri_betas = backend.asarray(train_fmri_betas)
                                train_latents = backend.asarray(train_latents)

                                start = time.time()
                                model.fit(train_latents, train_fmri_betas)
                                print("Best alphas: ", model.best_alphas_)

                                # pairwise_acc_scorer = make_scorer(pairwise_accuracy, greater_is_better=True)
                                # clf = GridSearchCV(model, param_grid={"alpha": args.l2_regularization_alphas},
                                #                    scoring=pairwise_acc_scorer, cv=NUM_CV_SPLITS, n_jobs=args.n_jobs,
                                #                    pre_dispatch=args.n_pre_dispatch_jobs, refit=True, verbose=3)
                                # clf.fit(train_fmri_betas, train_latents)
                                end = time.time()
                                print(f"Elapsed time: {int(end - start)}s")

                                best_alpha = model.best_alphas_

                                test_data_latents, _ = get_nn_latent_data(model_name, test_features,
                                                                          vision_features,
                                                                          lang_features,
                                                                          test_stim_ids,
                                                                          test_stim_types,
                                                                          subject,
                                                                          TESTING_MODE,
                                                                          nn_latent_transform=latent_transform)

                                imagery_data_latents, _ = get_nn_latent_data(model_name, features, vision_features,
                                                                             lang_features,
                                                                             imagery_stim_ids,
                                                                             imagery_stim_types,
                                                                             subject,
                                                                             IMAGERY,
                                                                             nn_latent_transform=latent_transform)

                                test_predicted_betas = model.predict(test_data_latents)
                                # imagery_predicted_latents = best_model.predict(imagery_fmri_betas)
                                corrs = []
                                for i in trange(test_predicted_betas.shape[1]):
                                    corr = pearsonr(test_predicted_betas[:, i], test_fmri_betas[:, i])[0]
                                    corrs.append(corr)
                                print("Mean corr: ", np.mean(corrs))
                                results = {
                                    "alpha": best_alpha,
                                    "model": model_name,
                                    "subject": subject,
                                    "features": features,
                                    "test_features": test_features,
                                    "vision_features": vision_features,
                                    "lang_features": lang_features,
                                    "training_mode": training_mode,
                                    "mask": mask,
                                    "num_voxels": test_fmri_betas.shape[1],
                                    # "cv_results": clf.cv_results_,
                                    "stimulus_ids": test_stim_ids,
                                    "stimulus_types": test_stim_types,
                                    "imagery_stimulus_ids": imagery_stim_ids,
                                    "predictions": test_predicted_betas,
                                    # "imagery_predictions": imagery_predicted_latents,
                                    "latents": test_data_latents,
                                    "imagery_latents": imagery_data_latents,
                                    "surface": args.surface,
                                    "resolution": args.resolution,
                                }

                                results.update(
                                    calc_all_pairwise_accuracy_scores(
                                        test_data_latents, test_predicted_latents, test_stim_types,
                                        imagery_data_latents, imagery_predicted_latents
                                    )
                                )
                                print(
                                    f"Best alpha: {best_alpha}"
                                    f" | Pairwise acc (mod-agnostic): {results[ACC_MODALITY_AGNOSTIC]:.2f}"
                                    f" | Pairwise acc (captions): {results[ACC_CAPTIONS]:.2f}"
                                    f" | Pairwise acc (images): {results[ACC_IMAGES]:.2f}"
                                    f" | Pairwise acc (imagery): {results[ACC_IMAGERY]:.2f}"
                                    f" | Pairwise acc (imagery whole test set): {results[ACC_IMAGERY_WHOLE_TEST]:.2f}"
                                )

                                os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                                pickle.dump(results, open(results_file_path, 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--surface", action="store_true", default=False)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--models", type=str, nargs='+', default=['imagebind'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--vision-features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--masks", type=str, nargs='+', default=[None])

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+',
                        default=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-pre-dispatch-jobs", type=int, default=DEFAULT_N_PRE_DISPATCH)

    parser.add_argument("--overwrite", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(ENCODER_OUT_DIR, exist_ok=True)

    run(args)
