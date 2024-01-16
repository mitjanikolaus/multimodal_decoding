############################################
# Training multimodal linear brain decoders
# inputs can be of any modality
# outputs are uni-modal
############################################
import argparse
import math
import time

import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist, cosine
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import os
from glob import glob
import pickle
from torchvision.transforms import Compose
from decoding_utils import get_distance_matrix
from tqdm import trange

from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, model_features_file_path, VISION_FEAT_KEY, LANG_FEAT_KEY

CONCAT_FEATS = 'concat'
AVG_FEATS = 'avg'
LANG_FEATS_ONLY = 'lang'
VISION_FEATS_ONLY = 'vision'
FEATURE_COMBINATION_CHOICES = [CONCAT_FEATS, AVG_FEATS, LANG_FEATS_ONLY, VISION_FEATS_ONLY]

NUM_CV_SPLITS = 5

TRAINING_MODES = ['train', 'train_captions', 'train_images']
DECODER_TESTING_MODES = ['test', 'test_captions', 'test_images']

MAX_EPOCHS = 400
BATCH_SIZE = 2000

GLM_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/glm/")
DISTANCE_METRICS = ['cosine']

SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']

MODEL_NAMES = ['VisualBERT', 'ImageBind', 'CLIP', 'BERT', 'GPT2_XL', 'VITL16', 'RESNET152', 'VILT']


class COCOBOLDDataset(Dataset):
    r"""
    Dataset for loading SEMREPS BOLD signals
    """

    def __init__(self, bold_root_dir, subject, model_name, mean_std_dir, mode=TRAINING_MODES[0], features=AVG_FEATS,
                 blank_correction=True, subset=None, fold=None, preloaded_betas=None,
                 overwrite_transformations_mean_std=False,
                 fmri_betas_transform=None, nn_latent_transform=None):
        """
        Args:
            bold_root_dir (str): BOLD root directory (parent of subjects' directories).
            subject (str): Subject ID.
            model_name (str): model name
            mode (str): 'train', 'test', or 'imagery'. You can append _images or _captions to make it unimodal
            blank_correction (boolean): If `True`, the blank image will be subtracted from the imagery patterns (if exists)
        """
        self.root_dir = os.path.join(bold_root_dir, subject)
        self.subject = subject
        self.imagery_scenes = IMAGERY_SCENES[subject]
        self.mode = mode
        self.features = features
        self.blank = None
        self.blank_correction = blank_correction
        self.mean_std_dir = mean_std_dir
        self.subset = subset
        self.overwrite_transformations_mean_std = overwrite_transformations_mean_std
        self.model_name = model_name
        self.fold = fold

        latent_vectors_file = model_features_file_path(model_name)

        latent_vectors = pickle.load(open(latent_vectors_file, 'rb'))

        fmri_addresses_regex = os.path.join(self.root_dir, f'betas_{self.mode}*', '*.nii')
        self.fmri_betas_addresses = np.array(sorted(glob(fmri_addresses_regex)))
        self.stim_ids = []
        self.stim_types = []
        self.nn_latent_vectors = []

        for addr in self.fmri_betas_addresses:
            file_name = os.path.basename(addr)
            if 'I' in file_name:  # Image
                stim_id = int(file_name[file_name.find('I') + 1:-4])
                self.stim_types.append('image')
            elif 'C' in file_name:  # Caption
                stim_id = int(file_name[file_name.find('C') + 1:-4])
                self.stim_types.append('caption')
            else:  # imagery
                stim_id = int(file_name[file_name.find('.nii') - 1:-4])
                stim_id = self.imagery_scenes[stim_id - 1][1]
                self.stim_types.append('imagery')
            self.stim_ids.append(stim_id)

            if self.features == VISION_FEATS_ONLY:
                feats = latent_vectors[stim_id][VISION_FEAT_KEY]
            elif self.features == LANG_FEATS_ONLY:
                feats = latent_vectors[stim_id][LANG_FEAT_KEY]
            elif self.features == AVG_FEATS:
                feats = np.stack((latent_vectors[stim_id][LANG_FEAT_KEY], latent_vectors[stim_id][VISION_FEAT_KEY]))
                feats = feats.mean(axis=0)
            elif self.features == CONCAT_FEATS:
                feats = np.concatenate(
                    (latent_vectors[stim_id][LANG_FEAT_KEY], latent_vectors[stim_id][VISION_FEAT_KEY]))
            else:
                raise RuntimeError(f"Unknown feature selection/combination method: {self.features}")
            self.nn_latent_vectors.append(feats)

        self.nn_latent_vectors = np.array(self.nn_latent_vectors, dtype=np.float32)
        self.stim_ids = np.array(self.stim_ids)
        self.stim_types = np.array(self.stim_types)

        if preloaded_betas is not None:
            assert len(preloaded_betas) == len(self.fmri_betas_addresses), f"Preloaded betas shape does not match!"
            self.fmri_betas = preloaded_betas
        else:
            self.fmri_betas = np.array([None for _ in range(len(self.fmri_betas_addresses))])

        brain_mask_address = os.path.join(self.root_dir, f'unstructured', 'mask.nii')
        self.brain_mask = nib.load(brain_mask_address).get_fdata().reshape(-1)
        self.brain_mask = np.logical_and(np.logical_not(np.isnan(self.brain_mask)), self.brain_mask != 0)
        self.bold_dim_size = self.brain_mask.sum()
        self.latent_dim_size = self.nn_latent_vectors[0].shape[0]

        beta_blank_address = os.path.join(self.root_dir, f'betas_blank', 'beta_blank.nii')
        if os.path.exists(beta_blank_address):
            self.blank = nib.load(beta_blank_address).get_fdata().astype('float32').reshape(-1)
            self.blank = self.blank[self.brain_mask]

        if self.subset is not None:
            self.fmri_betas_addresses = self.fmri_betas_addresses[subset]
            self.fmri_betas = self.fmri_betas[subset]
            self.nn_latent_vectors = self.nn_latent_vectors[subset]
            self.stim_ids = self.stim_ids[subset]
            self.stim_types = self.stim_types[subset]

        self.fmri_betas_transform = fmri_betas_transform
        self.nn_latent_transform = nn_latent_transform

        if self.fmri_betas_transform is None:
            self.init_fmri_betas_transform()

        if self.nn_latent_transform is None:
            self.init_nn_latent_transform()

    def init_nn_latent_transform(self):
        model_std_mean_name = f'{self.model_name}_{self.features}_mean_std_{self.mode}_fold_{self.fold}.p'
        model_std_mean_path = os.path.join(self.mean_std_dir, model_std_mean_name)
        if self.overwrite_transformations_mean_std or (not os.path.exists(model_std_mean_path)):
            print(f"Calculating Mean and STD of Model Latent Variables for {self.mode} samples (fold: {self.fold})")
            os.makedirs(self.mean_std_dir, exist_ok=True)

            mean_std = {'mean': self.nn_latent_vectors.mean(axis=0),
                        'std': self.nn_latent_vectors.std(axis=0)}
            pickle.dump(mean_std, open(model_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        model_mean_std = pickle.load(open(model_std_mean_path, 'rb'))
        compose_ops = [Normalize(model_mean_std['mean'], model_mean_std['std'])]
        self.nn_latent_transform = Compose(compose_ops)

    def init_fmri_betas_transform(self):
        bold_std_mean_name = f'bold_multimodal_mean_std_{self.mode}_fold_{self.fold}.p'
        bold_std_mean_path = os.path.join(self.mean_std_dir, bold_std_mean_name)

        if self.overwrite_transformations_mean_std or (not os.path.exists(bold_std_mean_path)):
            print(f"Calculating Mean and STD of BOLD Signals for {self.mode} samples (fold: {self.fold})")
            os.makedirs(self.mean_std_dir, exist_ok=True)
            self.preload()

            mean_std = {'mean': self.fmri_betas.mean(axis=0),
                        'std': self.fmri_betas.std(axis=0)}
            pickle.dump(mean_std, open(bold_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        bold_mean_std = pickle.load(open(bold_std_mean_path, 'rb'))
        compose_ops = [Normalize(bold_mean_std['mean'], bold_mean_std['std'])]
        self.fmri_betas_transform = Compose(compose_ops)

    def preload(self):
        print("preloading")
        for idx in trange(len(self.fmri_betas_addresses)):
            if self.fmri_betas[idx] is None:
                sample = nib.load(self.fmri_betas_addresses[idx]).get_fdata().astype('float32').reshape(-1)
                sample = sample[self.brain_mask]
                self.fmri_betas[idx] = sample.copy()
        return self.fmri_betas

    def get_latent_vector(self, idx):
        latent = self.nn_latent_vectors[idx]
        if self.nn_latent_transform is not None:
            latent = self.nn_latent_transform(latent)
        return latent

    def get_brain_vector(self, idx):
        if self.fmri_betas[idx] is not None:
            sample = self.fmri_betas[idx]
        else:
            sample = nib.load(self.fmri_betas_addresses[idx]).get_fdata().astype('float32').reshape(-1)
            sample = sample[self.brain_mask]
            self.fmri_betas[idx] = sample.copy()

        if self.mode == 'imagery' and self.blank is not None and self.blank_correction:
            sample = sample - self.blank
        if self.fmri_betas_transform is not None:
            sample = self.fmri_betas_transform(sample)
        return sample

    def get_stim_id(self, idx):
        sid = self.stim_ids[idx]
        return sid

    def get_stim_type(self, idx):
        return self.stim_types[idx]

    def __len__(self):
        return len(self.fmri_betas_addresses)

    def __getitem__(self, idx):
        sample = self.get_brain_vector(idx)
        sid = self.get_stim_id(idx)
        latent_vector = self.get_latent_vector(idx)
        stype = self.get_stim_type(idx)

        return sample, latent_vector, sid, stype


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).astype(np.float32).squeeze()


def calc_cosine_distances(predictions, targets):
    return np.mean([cosine(pred, target) for pred, target in zip(predictions, targets)])


def get_distance_matrix_csls(predictions, latents, knn=100, metric="cosine"):
    def get_nn_avg_dist(lat1, lat2, knn=10, metric="cosine"):
        distances = cdist(lat2, lat1, metric=metric)

        best_distances_idx = np.argsort(distances, axis=1)[:, -knn:]
        best_distances = distances[best_distances_idx]

        all_distances = best_distances.mean(axis=1)

        return all_distances

    average_dist_preds = get_nn_avg_dist(predictions, latents, knn, metric)
    average_dist_lats = get_nn_avg_dist(latents, predictions, knn, metric)

    scores = cdist(predictions, latents, metric=metric)

    dist_mat = 2 * scores - average_dist_preds - average_dist_lats

    return dist_mat


def pairwise_accuracy(predictions, latents, stimulus_ids, metric="cosine"):
    std = predictions.std(axis=0) + 1e-8  # For numerical stability
    predictions = (predictions - predictions.mean(axis=0)) / std

    if "csls_" in metric:
        metric = metric.replace("csls_", "")
        dist_mat = get_distance_matrix_csls(predictions, latents, metric=metric)
    else:
        dist_mat = get_distance_matrix(predictions, latents, metric)

    not_same_id = cdist(stimulus_ids.reshape(-1, 1), stimulus_ids.reshape(-1, 1)) != 0

    diag = dist_mat.diagonal().reshape(-1, 1)  # all congruent distances
    comp_mat = diag < dist_mat  # we are interested in i,j where d(i,i) < d(i,j)

    # Take only cases where the stimulus ids are not the same (do not compare cases where caption id == image id)
    comp_mat = comp_mat[not_same_id]
    score = comp_mat.mean()

    return score


def create_dissimilarity_matrix(sample_embeds):
    sim_mat = spearmanr(sample_embeds, axis=1)[0]
    dissim_mat = np.ones(sim_mat.shape) - sim_mat
    matrix = dissim_mat[np.triu_indices(sample_embeds.shape[0], 1)].reshape(-1)
    return matrix


def calc_rsa(latent_1, latent_2):
    matrix_1 = create_dissimilarity_matrix(latent_1)
    matrix_2 = create_dissimilarity_matrix(latent_2)
    corr = spearmanr([matrix_1, matrix_2], axis=1)[0]
    return corr


def calculate_eval_metrics(results, args):
    # take equally sized subsets of samples for captions and images
    stimulus_ids_caption = results["stimulus_ids"][results["stimulus_types"] == 'caption'][
                           :args.max_samples_eval_metrics]
    stimulus_ids_image = results["stimulus_ids"][results["stimulus_types"] != 'caption'][:args.max_samples_eval_metrics]
    val_ids = np.concatenate((stimulus_ids_caption, stimulus_ids_image))

    predictions_caption = results["predictions"][results["stimulus_types"] == 'caption'][:args.max_samples_eval_metrics]
    predictions_image = results["predictions"][results["stimulus_types"] != 'caption'][:args.max_samples_eval_metrics]
    val_predictions = np.concatenate((predictions_caption, predictions_image))

    latents_caption = results["latents"][results["stimulus_types"] == 'caption'][:args.max_samples_eval_metrics]
    latents_image = results["latents"][results["stimulus_types"] != 'caption'][:args.max_samples_eval_metrics]
    val_latents = np.concatenate((latents_caption, latents_image))

    for metric in DISTANCE_METRICS:
        acc = pairwise_accuracy(val_predictions, val_latents, val_ids, metric)
        results[f"acc_{metric}"] = acc

        acc_captions = pairwise_accuracy(predictions_caption, latents_caption, stimulus_ids_caption, metric)
        acc_images = pairwise_accuracy(predictions_image, latents_image, stimulus_ids_image, metric)
        results[f"acc_{metric}_captions"] = acc_captions
        results[f"acc_{metric}_images"] = acc_images

    rsa = calc_rsa(val_predictions, val_latents)
    results['rsa'] = rsa

    return results


def get_run_str(alpha, model_name, features, fold=None, best_val_loss=False, best_val_acc=False):
    run_str = f"{model_name}_{features}_alpha_{alpha}"
    if fold is not None:
        run_str += f"_fold_{fold}"
    if best_val_loss:
        run_str += "_best_val_loss"
    if best_val_acc:
        run_str += "_best_val_acc"
    return run_str


def train_and_test(alpha, results_dir, run_str, args, model_name, subject, train_ds, fold=None, val_ds=None, test_ds=None, best_val_loss=False, best_val_acc=False):
    results_file_dir = f'{results_dir}/{run_str}'
    os.makedirs(results_file_dir, exist_ok=True)

    train_data = [d for d in train_ds]
    train_data_inputs = np.array([i for i, _, _, _ in train_data])
    train_data_latents = np.array([l for _, l, _, _ in train_data])

    model = Ridge(alpha=alpha)
    model.fit(train_data_inputs, train_data_latents)

    results = {
        "alpha": alpha,
        "model": model_name,
        "subject": subject,
        "features": args.features,
        "training_mode": args.training_mode,
        "testing_mode": args.testing_mode,
        "fold": fold,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    if val_ds is not None:
        val_data = [d for d in val_ds]
        val_data_inputs = np.array([i for i, _, _, _ in val_data])
        val_data_latents = np.array([l for _, l, _, _ in val_data])

        val_predicted_latents = model.predict(val_data_inputs)
        val_loss = calc_cosine_distances(val_predicted_latents, val_data_latents)
        val_results = {"loss": val_loss,
                       "stimulus_ids": val_ds.stim_ids,
                       "stimulus_types": val_ds.stim_types,
                       "predictions": val_predicted_latents,
                       "latents": val_data_latents}
        val_results = calculate_eval_metrics(val_results, args)
        results = results | {'val_' + key: val for key, val in val_results.items()}

    if test_ds is not None:
        test_data = [d for d in test_ds]
        test_data_inputs = np.array([i for i, _, _, _ in test_data])
        test_data_latents = np.array([l for _, l, _, _ in test_data])
        test_predicted_latents = model.predict(test_data_inputs)

        test_results = {"stimulus_ids": test_ds.stim_ids,
                        "stimulus_types": test_ds.stim_types,
                        "predictions": test_predicted_latents,
                        "latents": test_data_latents}
        test_results = calculate_eval_metrics(test_results, args)
        results = results | test_results

    pickle.dump(results, open(os.path.join(results_file_dir, "results.p"), 'wb'))

    return results


def retrain_full_train(run_str, train_dataset, test_dataset, alpha, results_dir, args, model_name, subject, best_val_loss=False, best_val_acc=False):
    print(f"Retraining on full train set with alpha: ", alpha)

    train_and_test(alpha, results_dir, run_str, args, model_name, subject, train_dataset, val_ds=None, test_ds=test_dataset, best_val_loss=best_val_loss, best_val_acc=best_val_acc)


def run(args):
    for subject in args.subjects:
        print(subject)
        for model_name in args.models:
            model_name = model_name.lower()
            print(model_name)
            results_dir = os.path.join(GLM_OUT_DIR, f'{args.training_mode}/{subject}/')

            std_mean_dir = os.path.join(GLM_OUT_DIR, subject)

            train_val_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                args.training_mode, args.features)
            preloaded_betas = train_val_dataset.preload()

            kf = KFold(n_splits=NUM_CV_SPLITS, shuffle=True, random_state=1)

            test_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                           args.testing_mode, args.features,
                                           fmri_betas_transform=train_val_dataset.fmri_betas_transform,
                                           nn_latent_transform=train_val_dataset.nn_latent_transform)
            test_dataset.preload()

            best_alpha = None
            best_alpha_val_loss = math.inf

            best_alpha_pairwise_acc = None
            best_alpha_acc_cosine_value = 0
            for alpha in args.l2_regularization_alphas:
                print(f"alpha: {alpha}")

                val_losses_for_folds = []
                accs_cosine_for_folds = []

                start = time.time()

                for fold, (train_idx, val_idx) in enumerate(kf.split(list(range(len(train_val_dataset))))):
                    train_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                    args.training_mode, args.features, subset=train_idx, fold=fold,
                                                    preloaded_betas=preloaded_betas)
                    val_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                  args.training_mode, args.features, subset=val_idx, fold=fold,
                                                  preloaded_betas=preloaded_betas,
                                                  fmri_betas_transform=train_dataset.fmri_betas_transform,
                                                  nn_latent_transform=train_dataset.nn_latent_transform)
                    print(f"Fold {fold} | train set size: {len(train_dataset)} | val set size: {len(val_dataset)}")

                    run_str = get_run_str(alpha, model_name, args.features, fold=fold)

                    results = train_and_test(alpha, results_dir, run_str, args, model_name, subject, train_dataset, fold, val_dataset, test_dataset)

                    val_losses_for_folds.append(results['val_loss'])
                    print(f"best val loss: {results['val_loss']:.4f}")

                    accs_cosine_for_folds.append(results['val_acc_cosine'])
                    print(f"best val acc cosine: {results['val_acc_cosine']:.4f}")

                if np.mean(val_losses_for_folds) < best_alpha_val_loss:
                    best_alpha_val_loss = np.mean(val_losses_for_folds)
                    best_alpha = alpha
                    print(f"new best val loss: {np.mean(val_losses_for_folds):.4f}")

                if np.mean(accs_cosine_for_folds) > best_alpha_acc_cosine_value:
                    best_alpha_acc_cosine_value = np.mean(accs_cosine_for_folds)
                    best_alpha_pairwise_acc = alpha
                    print(f"new best pairwise acc: {np.mean(accs_cosine_for_folds):.4f}")

                end = time.time()
                print(f"Elapsed time: {int(end - start)}s")

            # Re-train on full train set with best HP settings:
            run_str = get_run_str(best_alpha, model_name, args.features, fold=None, best_val_loss=True)
            retrain_full_train(run_str, train_val_dataset, test_dataset, best_alpha, results_dir, args, model_name, subject, best_val_loss=True)

            run_str = get_run_str(best_alpha, model_name, args.features, fold=None, best_val_acc=True)
            retrain_full_train(run_str, train_val_dataset, test_dataset, best_alpha_pairwise_acc, results_dir, args, model_name, subject, best_val_acc=True)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-samples-eval-metrics", type=int, default=1000)

    parser.add_argument("--training-mode", type=str, default='train')
    parser.add_argument("--testing-mode", type=str, default='test')

    parser.add_argument("--models", type=str, nargs='+', default=['CLIP'])
    parser.add_argument("--features", type=str, default=CONCAT_FEATS, choices=FEATURE_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+', default=[1, 1e3, 1e5, 1e7])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(GLM_OUT_DIR, exist_ok=True)

    run(args)
