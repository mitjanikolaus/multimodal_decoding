############################################
# Training multimodal linear brain decoders
# inputs can be of any modality
# outputs are uni-modal
############################################
import argparse
import time

import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_destrieux_2009
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import os
from glob import glob
import pickle
from decoding_utils import get_distance_matrix
from tqdm import trange

from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, model_features_file_path, VISION_MEAN_FEAT_KEY, \
    VISION_CLS_FEAT_KEY, LANG_FEAT_KEY, \
    MULTIMODAL_FEAT_KEY

CONCAT_FEATS = 'concat'
AVG_FEATS = 'avg'
LANG_FEATS_ONLY = 'lang'
VISION_FEATS_ONLY = 'vision'
MULTIMODAL_FEATS = 'multi'
FEATS_SELECT_DEFAULT = 'default'
FEATURE_COMBINATION_CHOICES = [CONCAT_FEATS, AVG_FEATS, LANG_FEATS_ONLY, VISION_FEATS_ONLY, MULTIMODAL_FEATS,
                               FEATS_SELECT_DEFAULT]

VISION_CONCAT_FEATS = "concat"
VISION_FEAT_COMBINATION_CHOICES = [VISION_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY, VISION_CONCAT_FEATS]

NUM_CV_SPLITS = 5
DEFAULT_N_JOBS = 5
DEFAULT_N_PRE_DISPATCH = 5

DEFAULT_SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']

TRAIN_MODE_CHOICES = ["train", "train_captions", "train_images"]
TEST_MODE_CHOICES = ['test', 'test_captions', 'test_images']

GLM_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/glm/")
DISTANCE_METRICS = ['cosine']

MASK_ANATOMICAL_LANGUAGE = "anatomical_lang"

MASK_ANATOMICAL_ANGULAR_GYRUS = "anatomical_angular_gyrus"
MASK_ANATOMICAL_LEFT_ANGULAR_GYRUS = "anatomical_left_angular_gyrus"

MASK_ANATOMICAL_VISUAL_CORTEX = "anatomical_visual"
MASK_ANATOMICAL_VISUAL_CORTEX_OCCIPITAL = "anatomical_visual_occipital"
MASK_ANATOMICAL_VISUAL_CORTEX_V1 = "anatomical_visual_v1"

MASK_ANATOMICAL_OCCIPITAL_EXCLUSIVE = "anatomical_occipital_exclusive"

MASK_ANATOMICAL_VISUAL_HIGH_LEVEL = "anatomical_visual_high_level"

MASK_ANATOMICAL_TEMPORAL_CORTEX = "anatomical_temporal"
MASK_ANATOMICAL_TEMPORAL_CORTEX_EXCLUSIVE = "anatomical_temporal_exclusive"

MASK_ANATOMICAL_TEMPORAL_CORTEX_NOT_VISUAL = "anatomical_temporal_not_visual"

MASK_ANATOMICAL_NOT_VISUAL_CORTEX = "anatomical_not_visual"

REGIONS_OCCIPITAL_V1 = [
    'L G_occipital_middle', 'R G_occipital_middle', 'L S_oc_middle_and_Lunatus',
    'R S_oc_middle_and_Lunatus', 'L Pole_occipital', 'R Pole_occipital']
REGIONS_OCCIPITAL = [
    'L G_and_S_occipital_inf', 'L G_occipital_middle', 'L G_occipital_sup', 'L G_oc-temp_lat-fusifor',
    'L G_oc-temp_med-Lingual', 'L G_oc-temp_med-Parahip', 'L Pole_occipital',
    'L S_oc_middle_and_Lunatus', 'L S_oc_sup_and_transversal', 'L S_occipital_ant', 'L S_oc-temp_lat',
    'L S_oc-temp_med_and_Lingual', 'L S_parieto_occipital', 'R G_and_S_occipital_inf',
    'R G_occipital_middle', 'R G_occipital_sup', 'R G_oc-temp_lat-fusifor', 'R G_oc-temp_med-Lingual',
    'R G_oc-temp_med-Parahip', 'R Pole_occipital', 'R S_oc_middle_and_Lunatus',
    'R S_oc_sup_and_transversal', 'R S_occipital_ant', 'R S_oc-temp_lat',
    'R S_oc-temp_med_and_Lingual', 'R S_parieto_occipital']

REGIONS_OCCIPITAL_EXCLUSIVE = [
    'L G_and_S_occipital_inf',
    'L G_occipital_middle',
    'L G_occipital_sup',
    'L Pole_occipital',
    'L S_oc_middle_and_Lunatus',
    'L S_oc_sup_and_transversal',
    'L S_occipital_ant',
    'L S_parieto_occipital',
    'R G_and_S_occipital_inf',
    'R G_occipital_middle',
    'R G_occipital_sup',
    'R Pole_occipital',
    'R S_oc_middle_and_Lunatus',
    'R S_oc_sup_and_transversal',
    'R S_occipital_ant',
    'R S_parieto_occipital',
    'L G_oc-temp_med-Lingual',
    'R G_oc-temp_med-Lingual',
]

REGIONS_HIGH_LEVEL_VISUAL = [
    'L G_oc-temp_lat-fusifor',
    'R G_oc-temp_lat-fusifor',
    'L G_oc-temp_med-Parahip',
    'R G_oc-temp_med-Parahip',
    'L S_oc-temp_med_and_Lingual',
    'R S_oc-temp_med_and_Lingual',
    'L S_oc-temp_lat',
    'R S_oc-temp_lat',
    'L G_temporal_inf',
    'L G_temporal_middle',
    'L S_temporal_inf',
    'R G_temporal_inf',
    'R G_temporal_middle',
    'R S_temporal_inf',
]

REGIONS_TEMPORAL = [
    'L G_oc-temp_lat-fusifor',
    'L G_oc-temp_med-Lingual',
    'L G_oc-temp_med-Parahip',
    'L G_temp_sup-G_T_transv',
    'L G_temp_sup-Lateral',
    'L G_temp_sup-Plan_polar',
    'L G_temp_sup-Plan_tempo',
    'L G_temporal_inf',
    'L G_temporal_middle',
    'L Pole_temporal',
    'L S_oc-temp_lat',
    'L S_oc-temp_med_and_Lingual',
    'L S_temporal_inf',
    'L S_temporal_sup',
    'L S_temporal_transverse',
    'R G_oc-temp_lat-fusifor',
    'R G_oc-temp_med-Lingual',
    'R G_oc-temp_med-Parahip',
    'R G_temp_sup-G_T_transv',
    'R G_temp_sup-Lateral',
    'R G_temp_sup-Plan_polar',
    'R G_temp_sup-Plan_tempo',
    'R G_temporal_inf',
    'R G_temporal_middle',
    'R Pole_temporal',
    'R S_oc-temp_lat',
    'R S_oc-temp_med_and_Lingual',
    'R S_temporal_inf',
    'R S_temporal_sup',
    'R S_temporal_transverse']

REGIONS_TEMPORAL_EXCLUSIVE = [
    'L G_temp_sup-G_T_transv',
    'L G_temp_sup-Lateral',
    'L G_temp_sup-Plan_polar',
    'L G_temp_sup-Plan_tempo',
    'L G_temporal_inf',
    'L G_temporal_middle',
    'L Pole_temporal',
    'L S_temporal_inf',
    'L S_temporal_sup',
    'L S_temporal_transverse',
    'R G_temp_sup-G_T_transv',
    'R G_temp_sup-Lateral',
    'R G_temp_sup-Plan_polar',
    'R G_temp_sup-Plan_tempo',
    'R G_temporal_inf',
    'R G_temporal_middle',
    'R Pole_temporal',
    'R S_temporal_inf',
    'R S_temporal_sup',
    'R S_temporal_transverse']

REGIONS_LANGUAGE = [
    'L G_front_inf-Opercular',  # left inferior frontal gyrus
    'L G_front_inf-Orbital',  # left orbital inferior frontal gyrus
    'L G_front_inf-Triangul',  # left inferior frontal gyrus
    'L G_pariet_inf-Angular',  # left angular gyrus
    'L G_front_middle',  # left middle frontal gyrus
    'L G_front_sup',  # left superior frontal gyrus
    'L G_temp_sup-Lateral',  # lateral aspect of the superior temporal gyrus: middle-anterior temporal lobe?
    'L G_temp_sup-Plan_tempo',  # Planum temporale of the superior temporal gyrus
    'L G_temp_sup-Plan_polar',  # Planum polare of the superior temporal gyrus
    'L G_and_S_subcentral',  # Subcentral gyrus (central operculum) and sulci:
    'L Pole_temporal',  # Temporal pole
    'L G_pariet_inf-Supramar',  # Supramarginal gyrus:
    'L G_cingul-Post-dorsal',  # Posterior-dorsal part of the cingulate gyrus (dPCC)
    'L G_cingul-Post-ventral',  # Posterior-ventral part of the cingulate gyrus (vPCC)
]

REGIONS_ANGULAR_GYRUS = ['L G_pariet_inf-Angular', 'R G_pariet_inf-Angular']
REGIONS_LEFT_ANGULAR_GYRUS = ['L G_pariet_inf-Angular']


def get_roi_mask(roi_mask_name):
    destrieux_atlas = fetch_atlas_destrieux_2009()
    label_to_value_dict = {label[1]: int(label[0]) for label in destrieux_atlas['labels']}
    atlas_map = nib.load(destrieux_atlas.maps).get_fdata()

    if roi_mask_name == MASK_ANATOMICAL_VISUAL_CORTEX_OCCIPITAL:
        region_names = [label for label in REGIONS_OCCIPITAL]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    elif roi_mask_name == MASK_ANATOMICAL_VISUAL_CORTEX_V1:
        region_names = [label for label in REGIONS_OCCIPITAL_V1]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    elif roi_mask_name == MASK_ANATOMICAL_OCCIPITAL_EXCLUSIVE:
        region_names = [label for label in REGIONS_OCCIPITAL_EXCLUSIVE]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    elif roi_mask_name == MASK_ANATOMICAL_VISUAL_HIGH_LEVEL:
        region_names = [label for label in REGIONS_HIGH_LEVEL_VISUAL]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    elif roi_mask_name == MASK_ANATOMICAL_TEMPORAL_CORTEX:
        region_names = [label for label in REGIONS_TEMPORAL]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    elif roi_mask_name == MASK_ANATOMICAL_TEMPORAL_CORTEX_EXCLUSIVE:
        region_names = [label for label in REGIONS_TEMPORAL_EXCLUSIVE]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    elif roi_mask_name == MASK_ANATOMICAL_LANGUAGE:
        region_names = [label for label in REGIONS_LANGUAGE]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    elif roi_mask_name == MASK_ANATOMICAL_ANGULAR_GYRUS:
        region_names = [label for label in REGIONS_ANGULAR_GYRUS]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    elif roi_mask_name == MASK_ANATOMICAL_LEFT_ANGULAR_GYRUS:
        region_names = [label for label in REGIONS_LEFT_ANGULAR_GYRUS]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)

    else:
        raise RuntimeError("Unknown mask: ", roi_mask_name)

    return roi_mask


def get_default_features(model_name):
    if model_name.startswith("bert") or model_name.startswith("gpt") or model_name.startswith(
            "llama") or model_name.startswith("mistral") or model_name.startswith("mixtral"):
        features = LANG_FEATS_ONLY
    elif model_name.startswith("resnet") or model_name.startswith("vit") or model_name.startswith("dino"):
        features = VISION_FEATS_ONLY
    elif model_name.startswith("visualbert") or model_name.startswith("lxmert") or model_name.startswith(
            "vilt") or model_name.startswith("clip") or model_name.startswith("imagebind") or model_name.startswith(
        "flava"):
        features = CONCAT_FEATS
    else:
        raise RuntimeError(f"Unknown default features for {model_name}")

    print(f"Selected default features for {model_name}: {features}")
    return features


def get_nn_latent_data(model_name, features, vision_features_mode, stim_ids, subject, mode, nn_latent_transform=None,
                       recompute_std_mean=False):
    latent_vectors_file = model_features_file_path(model_name)
    latent_vectors = pickle.load(open(latent_vectors_file, 'rb'))

    nn_latent_vectors = []
    for stim_id in stim_ids:
        if not features in [LANG_FEATS_ONLY, MULTIMODAL_FEATS]:
            if vision_features_mode == VISION_MEAN_FEAT_KEY:
                vision_feats = latent_vectors[stim_id][VISION_MEAN_FEAT_KEY]
            elif vision_features_mode == VISION_CLS_FEAT_KEY:
                vision_feats = latent_vectors[stim_id][VISION_CLS_FEAT_KEY]
            elif vision_features_mode == VISION_CONCAT_FEATS:
                vision_feats_mean = latent_vectors[stim_id][VISION_MEAN_FEAT_KEY]
                vision_feats_cls = latent_vectors[stim_id][VISION_CLS_FEAT_KEY]
                vision_feats = np.concatenate((vision_feats_mean, vision_feats_cls))
            else:
                raise RuntimeError("Unknown vision feature combination choice: ", vision_features_mode)
        if features == VISION_FEATS_ONLY:
            feats = vision_feats
        elif features == LANG_FEATS_ONLY:
            feats = latent_vectors[stim_id][LANG_FEAT_KEY]
        elif features == AVG_FEATS:
            feats = np.stack((latent_vectors[stim_id][LANG_FEAT_KEY], vision_feats))
            feats = feats.mean(axis=0)
        elif features == CONCAT_FEATS:
            feats = np.concatenate(
                (latent_vectors[stim_id][LANG_FEAT_KEY], vision_feats))
        elif features == MULTIMODAL_FEATS:
            feats = latent_vectors[stim_id][MULTIMODAL_FEAT_KEY]
        else:
            raise RuntimeError(f"Unknown feature selection/combination method: {features}")
        nn_latent_vectors.append(feats)
    nn_latent_vectors = np.array(nn_latent_vectors, dtype=np.float32)

    if nn_latent_transform is None:
        mean_std_dir = os.path.join(GLM_OUT_DIR, subject)
        model_std_mean_name = f'{model_name}_{features}_{vision_features_mode}_mean_std_{mode}.p'
        model_std_mean_path = os.path.join(mean_std_dir, model_std_mean_name)
        if not os.path.exists(model_std_mean_path) or recompute_std_mean:
            print(f"Calculating Mean and STD of Model Latent Variables for {mode} samples")
            os.makedirs(mean_std_dir, exist_ok=True)

            mean_std = {'mean': nn_latent_vectors.mean(axis=0),
                        'std': nn_latent_vectors.std(axis=0)}
            pickle.dump(mean_std, open(model_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        model_mean_std = pickle.load(open(model_std_mean_path, 'rb'))
        nn_latent_transform = Normalize(model_mean_std['mean'], model_mean_std['std'])

    nn_latent_vectors = np.array([nn_latent_transform(v) for v in nn_latent_vectors])

    return nn_latent_vectors, nn_latent_transform


def get_fmri_data(subject, mode, fmri_betas_transform=None, roi_mask_name=None, recompute_std_mean=False):
    imagery_scenes = IMAGERY_SCENES[subject]

    fmri_data_dir = os.path.join(TWO_STAGE_GLM_DATA_DIR, subject)
    fmri_addresses_regex = os.path.join(fmri_data_dir, f'betas_{mode}*', '*.nii')
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

    gray_matter_mask_address = os.path.join(fmri_data_dir, f'unstructured', 'mask.nii')
    gray_matter_mask_img = nib.load(gray_matter_mask_address)
    gray_matter_mask_ras = nib.as_closest_canonical(gray_matter_mask_img)
    gray_matter_mask_ras_data = gray_matter_mask_ras.get_fdata()
    gray_matter_mask = gray_matter_mask_ras_data == 1

    if roi_mask_name is not None:
        roi_mask = get_roi_mask(roi_mask_name)
        print(f"Applying ROI mask of size {roi_mask.sum()}")
        print(f"Overlap with gray matter mask: {(roi_mask & gray_matter_mask).sum()}")

    mask = gray_matter_mask
    if roi_mask_name is not None:
        mask = roi_mask & gray_matter_mask

    fmri_betas = np.array([None for _ in range(len(fmri_betas_addresses))])
    for idx in trange(len(fmri_betas_addresses), desc="loading fmri data"):
        sample = nib.load(fmri_betas_addresses[idx])
        sample = nib.as_closest_canonical(sample)
        sample = sample.get_fdata()
        sample = sample[mask].astype('float32').reshape(-1)
        fmri_betas[idx] = sample.copy()

    if fmri_betas_transform is None:
        mean_std_dir = os.path.join(GLM_OUT_DIR, subject)
        bold_std_mean_name = f'bold_multimodal_mean_std_{mode}.p'
        if mask is not None:
            bold_std_mean_name += f'_mask_{roi_mask_name}'
        bold_std_mean_path = os.path.join(mean_std_dir, bold_std_mean_name)

        if not os.path.exists(bold_std_mean_path) or recompute_std_mean:
            print(f"Calculating mean and std of BOLD Signals for mode {mode} with mask {roi_mask_name}")
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


def create_dissimilarity_matrix(sample_embeds, matrix_metric="spearmanr"):
    if matrix_metric == "spearmanr":
        sim_mat = spearmanr(sample_embeds, axis=1)[0]
    elif matrix_metric == "pearsonr":
        sim_mat = np.corrcoef(sample_embeds, rowvar=1)
    else:
        raise RuntimeError("Unknown metric: ", matrix_metric)
    dissim_mat = np.ones(sim_mat.shape) - sim_mat
    matrix = dissim_mat[np.triu_indices(sample_embeds.shape[0], 1)].reshape(-1)
    return matrix


def rsa_from_matrices(matrix_1, matrix_2, metric="spearmanr"):
    if metric == "spearmanr":
        corr = spearmanr([matrix_1, matrix_2], axis=1)[0]
    elif metric == "pearsonr":
        corr = pearsonr(matrix_1, matrix_2)[0]
    else:
        raise RuntimeError("Unknown metric: ", metric)
    return corr


def calc_rsa(latent_1, latent_2, metric="spearmanr", matrix_metric="spearmanr"):
    matrix_1 = create_dissimilarity_matrix(latent_1, matrix_metric)
    matrix_2 = create_dissimilarity_matrix(latent_2, matrix_metric)
    return rsa_from_matrices(matrix_1, matrix_2, metric=metric)


def calc_rsa_images(latent_1, latent_2, stimulus_types, metric="spearmanr", matrix_metric="spearmanr"):
    assert len(latent_1) == len(latent_2) == len(stimulus_types)
    latent_1_images = latent_1[stimulus_types == 'image']
    latent_2_images = latent_2[stimulus_types == 'image']
    return calc_rsa(latent_1_images, latent_2_images, metric, matrix_metric)


def calc_rsa_captions(latent_1, latent_2, stimulus_types, metric="spearmanr", matrix_metric="spearmanr"):
    assert len(latent_1) == len(latent_2) == len(stimulus_types)
    latent_1_captions = latent_1[stimulus_types == 'caption']
    latent_2_captions = latent_2[stimulus_types == 'caption']
    return calc_rsa(latent_1_captions, latent_2_captions, metric, matrix_metric)


def calculate_eval_metrics(results, fmri_betas):
    # take equally sized subsets of samples for captions and images
    stimulus_ids_caption = results["stimulus_ids"][results["stimulus_types"] == 'caption']
    stimulus_ids_image = results["stimulus_ids"][results["stimulus_types"] != 'caption']
    val_ids = np.concatenate((stimulus_ids_caption, stimulus_ids_image))

    predictions_caption = results["predictions"][results["stimulus_types"] == 'caption']
    predictions_image = results["predictions"][results["stimulus_types"] != 'caption']
    val_predictions = np.concatenate((predictions_caption, predictions_image))

    latents_caption = results["latents"][results["stimulus_types"] == 'caption']
    latents_image = results["latents"][results["stimulus_types"] != 'caption']
    val_latents = np.concatenate((latents_caption, latents_image))

    fmri_betas_caption = fmri_betas[results["stimulus_types"] == 'caption']
    fmri_betas_image = fmri_betas[results["stimulus_types"] != 'caption']
    fmri_betas = np.concatenate((fmri_betas_caption, fmri_betas_image))

    for metric in DISTANCE_METRICS:
        acc = pairwise_accuracy(val_latents, val_predictions, val_ids, metric)
        results[f"acc_{metric}"] = acc

        acc_captions = pairwise_accuracy(latents_caption, predictions_caption, stimulus_ids_caption, metric)
        acc_images = pairwise_accuracy(latents_image, predictions_image, stimulus_ids_image, metric)
        results[f"acc_{metric}_captions"] = acc_captions
        results[f"acc_{metric}_images"] = acc_images

    results['rsa'] = calc_rsa(fmri_betas, val_latents)
    results['rsa_images'] = calc_rsa_images(fmri_betas, val_latents, results["stimulus_types"])
    results['rsa_captions'] = calc_rsa_captions(fmri_betas, val_latents, results["stimulus_types"])

    return results


def get_run_str(model_name, features, mask=None, best_val_loss=False, best_val_acc=False):
    run_str = f"{model_name}_{features}"
    if mask is not None:
        run_str += f"_mask_{mask}"
    if best_val_loss:
        run_str += "_best_val_loss"
    if best_val_acc:
        run_str += "_best_val_acc"
    return run_str


def run(args):
    for training_mode in args.training_modes:
        for mask in args.masks:
            for subject in args.subjects:
                train_fmri_betas, train_stim_ids, train_stim_types, fmri_transform = get_fmri_data(subject,
                                                                                                   training_mode,
                                                                                                   roi_mask_name=mask,
                                                                                                   recompute_std_mean=args.recompute_std_mean)
                test_fmri_betas, test_stim_ids, test_stim_types, _ = get_fmri_data(subject, args.testing_mode,
                                                                                   fmri_transform, roi_mask_name=mask)

                for model_name in args.models:
                    model_name = model_name.lower()

                    for features in args.features:
                        if features == FEATS_SELECT_DEFAULT:
                            features = get_default_features(model_name)
                        print(f"TRAIN MODE: {training_mode} | MASK: {mask} | SUBJECT: {subject} | "
                              f"MODEL: {model_name} | FEATURES: {features}")

                        train_data_latents, nn_latent_transform = get_nn_latent_data(model_name, features,
                                                                                     args.vision_features,
                                                                                     train_stim_ids,
                                                                                     subject,
                                                                                     training_mode,
                                                                                     recompute_std_mean=args.recompute_std_mean)

                        model = Ridge()
                        pairwise_acc_scorer = make_scorer(pairwise_accuracy, greater_is_better=True)
                        clf = GridSearchCV(model, param_grid={"alpha": args.l2_regularization_alphas},
                                           scoring=pairwise_acc_scorer, cv=NUM_CV_SPLITS, n_jobs=args.n_jobs,
                                           pre_dispatch=args.n_pre_dispatch_jobs, refit=True, verbose=3)

                        start = time.time()
                        clf.fit(train_fmri_betas, train_data_latents)
                        end = time.time()
                        print(f"Elapsed time: {int(end - start)}s")

                        best_alpha = clf.best_params_["alpha"]

                        results = {
                            "alpha": best_alpha,
                            "model": model_name,
                            "subject": subject,
                            "features": features,
                            "vision_features": args.vision_features,
                            "training_mode": training_mode,
                            "testing_mode": args.testing_mode,
                            "mask": mask,
                            "num_voxels": test_fmri_betas.shape[1],
                            "best_val_acc": True,
                            "cv_results": clf.cv_results_
                        }

                        test_data_latents, _ = get_nn_latent_data(model_name, features, args.vision_features,
                                                                  test_stim_ids,
                                                                  subject,
                                                                  args.testing_mode,
                                                                  nn_latent_transform=nn_latent_transform)
                        best_model = clf.best_estimator_
                        test_predicted_latents = best_model.predict(test_fmri_betas)

                        test_results = {"stimulus_ids": test_stim_ids,
                                        "stimulus_types": test_stim_types,
                                        "predictions": test_predicted_latents,
                                        "latents": test_data_latents}
                        test_results = calculate_eval_metrics(test_results, test_fmri_betas)
                        print(f"Best alpha: {best_alpha} | Pairwise acc: {test_results['acc_cosine']:.2f}"
                              f" | Pairwise acc (captions): {test_results['acc_cosine_captions']:.2f}"
                              f" | Pairwise acc (images): {test_results['acc_cosine_images']:.2f}"
                              f" | RSA (captions): {test_results['rsa_captions']:.2f}"
                              f" | RSA (images): {test_results['rsa_images']:.2f}")

                        results = results | test_results

                        results_dir = os.path.join(GLM_OUT_DIR, training_mode, subject)
                        run_str = get_run_str(model_name, features, mask, best_val_acc=True)
                        results_file_dir = f'{results_dir}/{run_str}'
                        os.makedirs(results_file_dir, exist_ok=True)

                        pickle.dump(results, open(os.path.join(results_file_dir, "results.p"), 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)
    parser.add_argument("--testing-mode", type=str, default='test', choices=TEST_MODE_CHOICES)

    parser.add_argument("--models", type=str, nargs='+', default=['CLIP'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=VISION_MEAN_FEAT_KEY,
                        choices=VISION_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--masks", type=str, nargs='+', default=[None])

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+', default=[1e3, 1e5, 1e7])

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-pre-dispatch-jobs", type=int, default=DEFAULT_N_PRE_DISPATCH)

    parser.add_argument("--recompute-std-mean", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(GLM_OUT_DIR, exist_ok=True)

    run(args)
