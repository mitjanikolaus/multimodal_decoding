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
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, GridSearchCV
from torch.utils.data import Dataset
import os
from glob import glob
import pickle
from torchvision.transforms import Compose
from decoding_utils import get_distance_matrix
from tqdm import trange

from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, model_features_file_path, VISION_FEAT_KEY, LANG_FEAT_KEY, \
    MULTIMODAL_FEAT_KEY

CONCAT_FEATS = 'concat'
AVG_FEATS = 'avg'
LANG_FEATS_ONLY = 'lang'
VISION_FEATS_ONLY = 'vision'
MULTIMODAL_FEATS = 'multi'
FEATURE_COMBINATION_CHOICES = [CONCAT_FEATS, AVG_FEATS, LANG_FEATS_ONLY, VISION_FEATS_ONLY, MULTIMODAL_FEATS]

NUM_CV_SPLITS = 5
N_JOBS = 8
PRE_DISPATCH = 8

TRAINING_MODES = ['train', 'train_captions', 'train_images']
DECODER_TESTING_MODES = ['test', 'test_captions', 'test_images']

GLM_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/glm/")
DISTANCE_METRICS = ['cosine']

SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']


def get_nn_latent_data(model_name, features, stim_ids, subject, mode, nn_latent_transform=None):
    latent_vectors_file = model_features_file_path(model_name)
    latent_vectors = pickle.load(open(latent_vectors_file, 'rb'))

    mean_std_dir = os.path.join(GLM_OUT_DIR, subject)

    nn_latent_vectors = []
    for stim_id in stim_ids:
        if features == VISION_FEATS_ONLY:
            feats = latent_vectors[stim_id][VISION_FEAT_KEY]
        elif features == LANG_FEATS_ONLY:
            feats = latent_vectors[stim_id][LANG_FEAT_KEY]
        elif features == AVG_FEATS:
            feats = np.stack((latent_vectors[stim_id][LANG_FEAT_KEY], latent_vectors[stim_id][VISION_FEAT_KEY]))
            feats = feats.mean(axis=0)
        elif features == CONCAT_FEATS:
            feats = np.concatenate(
                (latent_vectors[stim_id][LANG_FEAT_KEY], latent_vectors[stim_id][VISION_FEAT_KEY]))
        elif features == MULTIMODAL_FEATS:
            feats = latent_vectors[stim_id][MULTIMODAL_FEAT_KEY]
        else:
            raise RuntimeError(f"Unknown feature selection/combination method: {features}")
        nn_latent_vectors.append(feats)
    nn_latent_vectors = np.array(nn_latent_vectors, dtype=np.float32)

    if nn_latent_transform is None:
        model_std_mean_name = f'{model_name}_{features}_mean_std_{mode}.p'
        model_std_mean_path = os.path.join(mean_std_dir, model_std_mean_name)
        if not os.path.exists(model_std_mean_path):
            print(f"Calculating Mean and STD of Model Latent Variables for {mode} samples")
            os.makedirs(mean_std_dir, exist_ok=True)

            mean_std = {'mean': nn_latent_vectors.mean(axis=0),
                        'std': nn_latent_vectors.std(axis=0)}
            pickle.dump(mean_std, open(model_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        model_mean_std = pickle.load(open(model_std_mean_path, 'rb'))
        nn_latent_transform = Normalize(model_mean_std['mean'], model_mean_std['std'])

    nn_latent_vectors = np.array([nn_latent_transform(v) for v in nn_latent_vectors])

    return nn_latent_vectors, nn_latent_transform


def get_fmri_data(subject, mode=TRAINING_MODES[0], fmri_betas_transform=None):
    """
    Args:
        subject (str): Subject ID.
        mode (str): 'train', 'test', or 'imagery'. You can append _images or _captions to make it unimodal
        blank_correction (boolean): If `True`, the blank image will be subtracted from the imagery patterns (if exists)
    """
    bold_root_dir = os.path.join(TWO_STAGE_GLM_DATA_DIR, subject)
    imagery_scenes = IMAGERY_SCENES[subject]
    mean_std_dir = os.path.join(GLM_OUT_DIR, subject)

    fmri_addresses_regex = os.path.join(bold_root_dir, f'betas_{mode}*', '*.nii')
    fmri_betas_addresses = np.array(sorted(glob(fmri_addresses_regex)))
    stim_ids = []
    stim_types = []

    for addr in fmri_betas_addresses:
        file_name = os.path.basename(addr)
        if 'I' in file_name:  # Image
            stim_id = int(file_name[file_name.find('I') + 1:-4])
            stim_types.append('image')
        elif 'C' in file_name:  # Caption
            stim_id = int(file_name[file_name.find('C') + 1:-4])
            stim_types.append('caption')
        else:  # imagery
            stim_id = int(file_name[file_name.find('.nii') - 1:-4])
            stim_id = imagery_scenes[stim_id - 1][1]
            stim_types.append('imagery')
        stim_ids.append(stim_id)

    stim_ids = np.array(stim_ids)
    stim_types = np.array(stim_types)

    fmri_betas = np.array([None for _ in range(len(fmri_betas_addresses))])

    brain_mask_address = os.path.join(bold_root_dir, f'unstructured', 'mask.nii')
    brain_mask = nib.load(brain_mask_address).get_fdata().reshape(-1)
    brain_mask = np.logical_and(np.logical_not(np.isnan(brain_mask)), brain_mask != 0)

    for idx in trange(len(fmri_betas_addresses), desc="loading fmri data"):
        if fmri_betas[idx] is None:
            sample = nib.load(fmri_betas_addresses[idx]).get_fdata().astype('float32').reshape(-1)
            sample = sample[brain_mask]
            fmri_betas[idx] = sample.copy()

    if fmri_betas_transform is None:
        bold_std_mean_name = f'bold_multimodal_mean_std_{mode}.p'
        bold_std_mean_path = os.path.join(mean_std_dir, bold_std_mean_name)

        if not os.path.exists(bold_std_mean_path):
            print(f"Calculating Mean and STD of BOLD Signals for {mode} samples")
            os.makedirs(mean_std_dir, exist_ok=True)

            mean_std = {'mean': fmri_betas.mean(axis=0),
                        'std': fmri_betas.std(axis=0)}
            pickle.dump(mean_std, open(bold_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        bold_mean_std = pickle.load(open(bold_std_mean_path, 'rb'))
        fmri_betas_transform = Normalize(bold_mean_std['mean'], bold_mean_std['std'])

    fmri_betas = np.array([fmri_betas_transform(v) for v in fmri_betas])

    return fmri_betas, stim_ids, stim_types, fmri_betas_transform


class Normalize:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std
        self.std = self.std + eps  # Avoid division by 0

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


def pairwise_accuracy(latents, predictions, stimulus_ids=None, metric="cosine"):
    pred_normalize = Normalize(predictions.mean(axis=0), predictions.std(axis=0))
    predictions = pred_normalize(predictions)

    if "csls_" in metric:
        metric = metric.replace("csls_", "")
        dist_mat = get_distance_matrix_csls(predictions, latents, metric=metric)
    else:
        dist_mat = get_distance_matrix(predictions, latents, metric)

    diag = dist_mat.diagonal().reshape(-1, 1)  # all congruent distances
    comp_mat = diag < dist_mat  # we are interested in i,j where d(i,i) < d(i,j)

    if stimulus_ids is not None:
        # Take only cases where the stimulus ids are not the same (do not compare cases where caption id == image id)
        not_same_id = cdist(stimulus_ids.reshape(-1, 1), stimulus_ids.reshape(-1, 1)) != 0
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
        acc = pairwise_accuracy(val_latents, val_predictions, val_ids, metric)
        results[f"acc_{metric}"] = acc

        acc_captions = pairwise_accuracy(latents_caption, predictions_caption, stimulus_ids_caption, metric)
        acc_images = pairwise_accuracy(latents_image, predictions_image, stimulus_ids_image, metric)
        results[f"acc_{metric}_captions"] = acc_captions
        results[f"acc_{metric}_images"] = acc_images

    rsa = calc_rsa(val_predictions, val_latents)
    results['rsa'] = rsa

    return results


def get_run_str(alpha, model_name, features, fold=None, best_val_loss=False, best_val_acc=False):
    run_str = f"{model_name}_{features}"
    if not best_val_acc and not best_val_loss:
        run_str += f"_alpha_{alpha}"
    if fold is not None:
        run_str += f"_fold_{fold}"
    if best_val_loss:
        run_str += "_best_val_loss"
    if best_val_acc:
        run_str += "_best_val_acc"
    return run_str


def run(args):
    for features in args.features:
        print("FEATURES: ", features)
        for subject in args.subjects:
            print("SUBJECT: ", subject)
            fmri_betas, train_stim_ids, train_stim_types, fmri_transform = get_fmri_data(subject, args.training_mode)

            fmri_test_betas, test_stim_ids, test_stim_types, _ = get_fmri_data(subject, args.testing_mode,
                                                                               fmri_transform)

            for model_name in args.models:
                model_name = model_name.lower()
                print(model_name)
                results_dir = os.path.join(GLM_OUT_DIR, f'{args.training_mode}/{subject}/')

                train_data_inputs = fmri_betas
                train_data_latents, nn_latent_transform = get_nn_latent_data(model_name, features, train_stim_ids,
                                                                             subject,
                                                                             args.training_mode)

                model = Ridge()
                pairwise_acc_scorer = make_scorer(pairwise_accuracy, greater_is_better=True)

                clf = GridSearchCV(model, param_grid={"alpha": args.l2_regularization_alphas},
                                   scoring=pairwise_acc_scorer, cv=NUM_CV_SPLITS, n_jobs=N_JOBS,
                                   pre_dispatch=PRE_DISPATCH, refit=True, verbose=3)

                start = time.time()
                clf.fit(train_data_inputs, train_data_latents)

                end = time.time()
                print(f"Elapsed time: {int(end - start)}s")

                best_alpha = clf.best_params_["alpha"]

                results = {
                    "alpha": clf.best_params_["alpha"],
                    "model": model_name,
                    "subject": subject,
                    "features": features,
                    "training_mode": args.training_mode,
                    "testing_mode": args.testing_mode,
                    "best_val_acc": True,
                }

                best_model = clf.best_estimator_

                test_data_latents, _ = get_nn_latent_data(model_name, features, test_stim_ids, subject,
                                                          args.testing_mode,
                                                          nn_latent_transform=nn_latent_transform)
                test_predicted_latents = best_model.predict(fmri_test_betas)

                test_results = {"stimulus_ids": test_stim_ids,
                                "stimulus_types": test_stim_types,
                                "predictions": test_predicted_latents,
                                "latents": test_data_latents}
                test_results = calculate_eval_metrics(test_results, args)
                print(f"Best alpha: {best_alpha} | Pairwise acc: {test_results['acc_cosine']:.3f}")

                results = results | test_results

                run_str = get_run_str(best_alpha, model_name, features, fold=None, best_val_acc=True)
                results_file_dir = f'{results_dir}/{run_str}'
                os.makedirs(results_file_dir, exist_ok=True)

                pickle.dump(results, open(os.path.join(results_file_dir, "results.p"), 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-samples-eval-metrics", type=int, default=1000)

    parser.add_argument("--training-mode", type=str, default='train')
    parser.add_argument("--testing-mode", type=str, default='test')

    parser.add_argument("--models", type=str, nargs='+', default=['CLIP'])
    parser.add_argument("--features", type=str, nargs='+', default=[CONCAT_FEATS])

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+', default=[1e3, 1e5, 1e7])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(GLM_OUT_DIR, exist_ok=True)

    run(args)
