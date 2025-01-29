import argparse

import numpy as np
import torch.cuda
import os
import pickle

from data import fMRIDataModule, LatentFeatsConfig, TRAINING_MODES, MODALITY_AGNOSTIC, SELECT_DEFAULT
from new_decoding.decoder import Decoder, test_set_pairwise_acc_scores
from analyses.decoding.ridge_regression_decoding import FEATURE_COMBINATION_CHOICES, \
    VISION_FEAT_COMBINATION_CHOICES, LANG_FEAT_COMBINATION_CHOICES, get_run_str
from utils import SUBJECTS, RESULTS_FILE, PYTORCH_DECODER_OUT_DIR, FMRI_BETAS_DIR, DEFAULT_MODEL
import lightning as pl

DEFAULT_NUM_WORKERS = 10
DEFAULT_MAX_EPOCHS = 50
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_MSE_LOSS_WEIGHT = 0.5


def run(args):
    for training_mode in args.training_modes:
        for subject in args.subjects:
            for model_name in args.models:
                feats_config = LatentFeatsConfig(
                    model_name, args.features, args.test_features, args.vision_features, args.lang_features
                )

                print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                      f"MODEL: {model_name} | FEATURES: {feats_config.features} {feats_config.vision_features} "
                      f"{feats_config.lang_features} | TEST FEATURES: {feats_config.test_features}")

                run_str = get_run_str(args.betas_dir, feats_config)
                results_file_path = os.path.join(
                    PYTORCH_DECODER_OUT_DIR, training_mode, subject, run_str, RESULTS_FILE
                )
                if os.path.isfile(results_file_path) and not args.overwrite:
                    print(f"Skipping decoder training as results are already present at"
                          f" {results_file_path}")
                    continue

                num_cv_splits = 5
                test_set_preds = []
                test_targets = None
                test_stim_types = None
                for cv in range(num_cv_splits):
                    dm = fMRIDataModule(args.betas_dir, args.batch_size, subject, training_mode,
                                        feats_config, args.num_workers, cv_split=cv,
                                        num_cv_splits=num_cv_splits)

                    sample_betas, sample_latents = next(iter(dm.ds_train))

                    model = Decoder(sample_betas.size, sample_latents.size, args.learning_rate,
                                    args.weight_decay, args.batch_size, args.mse_loss_weight,
                                    cosine_loss=args.cosine_loss)

                    # Initialize wandb logger
                    # TODO
                    # wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

                    # Initialize Callbacks
                    early_stop_callback = pl.pytorch.callbacks.EarlyStopping(monitor="val_loss")
                    checkpoint_callback = pl.pytorch.callbacks.ModelCheckpoint(monitor="val_loss")
                    trainer = pl.Trainer(max_epochs=args.max_epochs,
                                         # logger=wandb_logger, #TODO
                                         callbacks=[early_stop_callback,
                                                    checkpoint_callback],
                                         log_every_n_steps=10,
                                         reload_dataloaders_every_n_epochs=1,
                                         precision='16-mixed',
                                         )

                    trainer.fit(model, dm)

                    trainer.test(dataloaders=dm.test_dataloader(), ckpt_path='best')
                    # Close wandb run
                    # wandb.finish() #TODO

                    test_set_preds.append(model.test_outputs['preds'])
                    test_targets = model.test_outputs['targets']
                    test_stim_types = np.array(model.test_outputs['stim_types'])

                    pickle.dump(model.test_outputs['preds'].cpu().numpy(), open('test_preds.p', 'wb'))

                mean_preds = torch.stack(test_set_preds).mean(dim=0).cpu().numpy()
                scores = test_set_pairwise_acc_scores(test_targets, mean_preds, test_stim_types)

                results = {
                    # "alpha": best_alpha,
                    "model": model_name,
                    "subject": subject,
                    # "features": features,
                    # "test_features": test_features,
                    # "vision_features": vision_features,
                    # "lang_features": lang_features,
                    "training_mode": training_mode,
                    # "num_voxels": test_fmri_betas.shape[1],
                    # "cv_results": clf.cv_results_,
                    # "stimulus_ids": test_stim_ids,
                    "stimulus_types": test_stim_types,
                    # "imagery_stimulus_ids": imagery_stim_ids,
                    "predictions": mean_preds,
                    # "imagery_predictions": imagery_predicted_latents,
                    "latents": test_targets,
                    # "imagery_latents": imagery_data_latents,
                }

                print("\n\n\n")
                print("Results with mean preds: ")
                for score, val in scores.items():
                    print(f"{score}: {val.cpu().numpy():.3f}")
                    results[score]: val.cpu().numpy()

                # results.update(
                #     calc_all_pairwise_accuracy_scores(
                #         test_data_latents, test_predicted_latents, test_stim_types,
                #         imagery_data_latents, imagery_predicted_latents
                #     )
                # )

                os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                pickle.dump(results, open(results_file_path, 'wb'))

                # print(
                #     f"Best alpha: {best_alpha}"
                #     f" | Pairwise acc (mod-agnostic): {results[ACC_MODALITY_AGNOSTIC]:.2f}"
                #     f" | Pairwise acc (captions): {results[ACC_CAPTIONS]:.2f}"
                #     f" | Pairwise acc (images): {results[ACC_IMAGES]:.2f}"
                #     f" | Pairwise acc (imagery): {results[ACC_IMAGERY]:.2f}"
                #     f" | Pairwise acc (imagery whole test set): {results[ACC_IMAGERY_WHOLE_TEST]:.2f}"
                # )
                #
                os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
                # pickle.dump(results, open(results_file_path, 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

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

    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--mse-loss-weight", type=float, default=DEFAULT_MSE_LOSS_WEIGHT)
    parser.add_argument("--cosine-loss", action='store_true', default=False)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+',
                        default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

    parser.add_argument("--overwrite", action='store_true', default=False)

    return parser.parse_args()


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    args = get_args()
    os.makedirs(PYTORCH_DECODER_OUT_DIR, exist_ok=True)

    run(args)
