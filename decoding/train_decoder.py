import argparse
import time

import torch.cuda
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import os
import pickle

from decoding.data import fMRIDataModule, LatentFeatsConfig, TRAIN_MODE_CHOICES, MODALITY_AGNOSTIC
from decoding.decoder import Decoder
from analyses.ridge_regression_decoding import FEATS_SELECT_DEFAULT, FEATURE_COMBINATION_CHOICES, \
    VISION_FEAT_COMBINATION_CHOICES, LANG_FEAT_COMBINATION_CHOICES, IMAGERY, TESTING_MODE, get_default_features, \
    get_default_vision_features, get_default_lang_features, get_run_str, get_nn_latent_data
from utils import SUBJECTS, ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, ACC_MODALITY_AGNOSTIC, \
    RESULTS_FILE, DECODER_OUT_DIR
import lightning as pl


DEFAULT_NUM_WORKERS = 10
DEFAULT_MAX_EPOCHS = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3


def run(args):
    for training_mode in args.training_modes:
        for subject in args.subjects:
            for model_name in args.models:
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

                            print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                                  f"MODEL: {model_name} | FEATURES: {features} {vision_features} {lang_features} | "
                                  f"TEST FEATURES: {test_features}")

                            run_str = get_run_str(
                                model_name, features, test_features, vision_features, lang_features, mask=None,
                                surface=False, resolution=None)
                            results_file_path = os.path.join(DECODER_OUT_DIR, training_mode, subject, run_str, RESULTS_FILE)
                            if os.path.isfile(results_file_path) and not args.overwrite:
                                print(f"Skipping decoder training as results are already present at"
                                      f" {results_file_path}")
                                continue

                            latent_feats_config = LatentFeatsConfig(
                                model_name, features, vision_features, lang_features
                            )
                            dm = fMRIDataModule(args.batch_size, subject, training_mode, latent_feats_config, args.num_workers)

                            sample_betas, sample_latents = next(iter(dm.ds_train))

                            model = Decoder(sample_betas.size, sample_latents.size, args.learning_rate, args.batch_size)

                            # Initialize wandb logger
                            #TODO
                            # wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

                            # Initialize Callbacks
                            early_stop_callback = pl.pytorch.callbacks.EarlyStopping(monitor="val_loss")
                            # checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint()
                            trainer = pl.Trainer(max_epochs=args.max_epochs,
                                                 # logger=wandb_logger, #TODO
                                                 callbacks=[early_stop_callback],
                                                    log_every_n_steps=10,
                                                 )

                            trainer.fit(model, dm)

                            trainer.test(dataloaders=dm.test_dataloader(), ckpt_path='best')

                            # Close wandb run
                            # wandb.finish() #TODO





                            pairwise_acc_scorer = make_scorer(pairwise_accuracy, greater_is_better=True)
                            clf = GridSearchCV(model, param_grid={"alpha": args.l2_regularization_alphas},
                                               scoring=pairwise_acc_scorer, cv=NUM_CV_SPLITS, n_jobs=args.n_jobs,
                                               pre_dispatch=args.n_pre_dispatch_jobs, refit=True, verbose=3)

                            start = time.time()
                            clf.fit(train_fmri_betas, train_latents)
                            end = time.time()
                            print(f"Elapsed time: {int(end - start)}s")

                            best_alpha = clf.best_params_["alpha"]

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

                            best_model = clf.best_estimator_
                            test_predicted_latents = best_model.predict(test_fmri_betas)
                            imagery_predicted_latents = best_model.predict(imagery_fmri_betas)

                            results = {
                                "alpha": best_alpha,
                                "model": model_name,
                                "subject": subject,
                                "features": features,
                                "test_features": test_features,
                                "vision_features": vision_features,
                                "lang_features": lang_features,
                                "training_mode": training_mode,
                                "num_voxels": test_fmri_betas.shape[1],
                                "cv_results": clf.cv_results_,
                                "stimulus_ids": test_stim_ids,
                                "stimulus_types": test_stim_types,
                                "imagery_stimulus_ids": imagery_stim_ids,
                                "predictions": test_predicted_latents,
                                "imagery_predictions": imagery_predicted_latents,
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

    parser.add_argument("--training-modes", type=str, nargs="+", default=[MODALITY_AGNOSTIC],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--models", type=str, nargs='+', default=['imagebind'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--test-features", type=str, default=FEATS_SELECT_DEFAULT,
                        choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--vision-features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=int, default=DEFAULT_LEARNING_RATE)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+',
                        default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

    parser.add_argument("--overwrite", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    args = get_args()
    os.makedirs(DECODER_OUT_DIR, exist_ok=True)

    run(args)
