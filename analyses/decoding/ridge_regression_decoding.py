import argparse
import time

from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import os
import pickle

from data import LatentFeatsConfig, SELECT_DEFAULT, FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, \
    LANG_FEAT_COMBINATION_CHOICES, get_fmri_data, apply_mask, standardize_fmri_betas, get_latent_features, \
    standardize_latents, TESTING_MODE, IMAGERY, remove_nans
from eval import pairwise_accuracy, calc_all_pairwise_accuracy_scores, ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, \
    ACC_IMAGERY_WHOLE_TEST
from utils import FMRI_BETAS_DIR, SUBJECTS, DEFAULT_RESOLUTION, RESULTS_FILE, MODE_AGNOSTIC, TRAIN_MODE_CHOICES, \
    RIDGE_DECODER_OUT_DIR

NUM_CV_SPLITS = 5
DEFAULT_N_JOBS = 5
DEFAULT_N_PRE_DISPATCH = 5

DEFAULT_ALPHAS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


def get_run_str(model_name, feats_config, mask, surface, resolution,
                hemi=None):
    run_str = f"{model_name}_{feats_config.combined_feats}"
    run_str += f"_{feats_config.vision_features}"
    run_str += f"_{feats_config.lang_features}"

    if mask is not None:
        if mask.startswith("functional_") or mask.startswith("anatomical_"):
            run_str += f"_mask_{mask}"
        elif "p_values" in mask:
            mask_name = os.path.basename(mask).replace(".p", "")
            run_str += f"_mask_{mask_name}"
        else:
            raise RuntimeError(f"Unsupported mask: {mask}")

    if surface:
        run_str += f"_surface_{resolution}"

    if hemi:
        run_str += f"_{hemi}_hemi"

    return run_str


def run(args):
    for training_mode in args.training_modes:
        for subject in args.subjects:
            train_fmri_betas_full, train_stim_ids, train_stim_types = get_fmri_data(
                args.betas_dir,
                subject,
                training_mode,
                surface=args.surface,
                resolution=args.resolution,
            )
            test_fmri_betas_full, test_stim_ids, test_stim_types = get_fmri_data(
                args.betas_dir,
                subject,
                TESTING_MODE,
                surface=args.surface,
                resolution=args.resolution,
            )
            imagery_fmri_betas_full, imagery_stim_ids, imagery_stim_types = get_fmri_data(
                args.betas_dir,
                subject,
                IMAGERY,
                surface=args.surface,
                resolution=args.resolution,
            )
            for mask in args.masks:
                mask = None if mask in ["none", "None"] else mask
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = apply_mask(
                    mask, [train_fmri_betas_full, test_fmri_betas_full, imagery_fmri_betas_full], args
                )
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = remove_nans(
                    [train_fmri_betas, test_fmri_betas, imagery_fmri_betas]
                )
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = standardize_fmri_betas(
                    train_fmri_betas, test_fmri_betas, imagery_fmri_betas,
                )

                for model in args.models:
                    feats_config = LatentFeatsConfig(
                        model, args.features, args.test_features, args.vision_features, args.lang_features
                    )

                    train_latents = get_latent_features(model, feats_config, train_stim_ids, train_stim_types)
                    test_latents = get_latent_features(
                        model, feats_config, test_stim_ids, test_stim_types, test_mode=True
                    )
                    imagery_latents = get_latent_features(
                        model, feats_config, imagery_stim_ids, imagery_stim_types, test_mode=True
                    )
                    train_latents, test_latents, imagery_latents = standardize_latents(
                        train_latents, test_latents, imagery_latents
                    )

                    print(f"\nTRAIN MODE: {training_mode} | MASK: {mask} | SUBJECT: {subject} | "
                          f"MODEL: {model} | FEATURES: {feats_config.features} {feats_config.vision_features} "
                          f"{feats_config.lang_features} | TEST FEATURES: {feats_config.test_features}")
                    print(f"train fMRI betas shape: {train_fmri_betas.shape}")
                    print(f"test fMRI betas shape: {test_fmri_betas.shape}")
                    print(f"imagery fMRI betas shape: {imagery_fmri_betas.shape}")
                    print(f"train latents shape: {train_latents.shape}")

                    run_str = get_run_str(model, feats_config, mask, args.surface, args.resolution)
                    results_file_path = os.path.join(
                        RIDGE_DECODER_OUT_DIR, training_mode, subject, run_str, RESULTS_FILE
                    )
                    if os.path.isfile(results_file_path) and not args.overwrite:
                        print(f"Skipping decoder training as results are already present at {results_file_path}")
                        continue

                    pairwise_acc_scorer = make_scorer(pairwise_accuracy, greater_is_better=True)
                    clf = GridSearchCV(
                        Ridge(), param_grid={"alpha": args.l2_regularization_alphas}, scoring=pairwise_acc_scorer,
                        cv=NUM_CV_SPLITS, n_jobs=args.n_jobs, pre_dispatch=args.n_pre_dispatch_jobs, refit=True,
                        verbose=3
                    )
                    start = time.time()
                    clf.fit(train_fmri_betas, train_latents)
                    end = time.time()
                    print(f"Elapsed time: {int(end - start)}s")

                    best_alpha = clf.best_params_["alpha"]

                    best_model = clf.best_estimator_

                    test_predicted_latents = best_model.predict(test_fmri_betas)
                    imagery_predicted_latents = best_model.predict(imagery_fmri_betas)

                    train_predicted_latents = best_model.predict(train_fmri_betas)
                    pickle.dump(train_predicted_latents, open(f"{subject}_train_preds.p", 'wb'))

                    results = {
                        "alpha": best_alpha,
                        "model": model,
                        "subject": subject,
                        "features": feats_config.features,
                        "test_features": feats_config.test_features,
                        "vision_features": feats_config.vision_features,
                        "lang_features": feats_config.lang_features,
                        "training_mode": training_mode,
                        "mask": mask,
                        "num_voxels": test_fmri_betas.shape[1],
                        "cv_results": clf.cv_results_,
                        "stimulus_ids": test_stim_ids,
                        "stimulus_types": test_stim_types,
                        "imagery_stimulus_ids": imagery_stim_ids,
                        "predictions": test_predicted_latents,
                        "imagery_predictions": imagery_predicted_latents,
                        "latents": test_latents,
                        "imagery_latents": imagery_latents,
                        "surface": args.surface,
                        "resolution": args.resolution,
                    }
                    scores = calc_all_pairwise_accuracy_scores(
                        test_latents, test_predicted_latents, test_stim_types,
                        imagery_latents, imagery_predicted_latents
                    )
                    results.update(scores)
                    print(
                        f"Best alpha: {best_alpha}"
                        f" | Pairwise acc (captions): {results[ACC_CAPTIONS]:.2f}"
                        f" | Pairwise acc (images): {results[ACC_IMAGES]:.2f}"
                        f" | Pairwise acc (imagery): {results[ACC_IMAGERY]:.2f}"
                        f" | Pairwise acc (imagery whole test set): {results[ACC_IMAGERY_WHOLE_TEST]:.2f}"
                    )

                    results_no_standardization = calc_all_pairwise_accuracy_scores(
                        test_latents, test_predicted_latents, test_stim_types,
                        imagery_latents, imagery_predicted_latents, standardize_predictions=False
                    )
                    print(
                        f" | Pairwise acc (no std) (captions): {results_no_standardization[ACC_CAPTIONS]:.2f}"
                        f" | Pairwise acc (no std) (images): {results_no_standardization[ACC_IMAGES]:.2f}"
                        f" | Pairwise acc (no std) (imagery): {results_no_standardization[ACC_IMAGERY]:.2f}"
                        f" | Pairwise acc (no std) (imagery whole test set): "
                        f"{results_no_standardization[ACC_IMAGERY_WHOLE_TEST]:.2f}"
                    )
                    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                    pickle.dump(results, open(results_file_path, 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--training-modes", type=str, nargs="+", default=[MODE_AGNOSTIC],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--surface", action="store_true", default=False)
    parser.add_argument("--resolution", type=str, default=DEFAULT_RESOLUTION)

    parser.add_argument("--models", type=str, nargs='+', default=['imagebind'])
    parser.add_argument("--features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--vision-features", type=str, default=SELECT_DEFAULT,
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, default=SELECT_DEFAULT,
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--masks", type=str, nargs='+', default=[None])

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+', default=DEFAULT_ALPHAS)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-pre-dispatch-jobs", type=int, default=DEFAULT_N_PRE_DISPATCH)

    parser.add_argument("--overwrite", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(RIDGE_DECODER_OUT_DIR, exist_ok=True)

    run(args)
