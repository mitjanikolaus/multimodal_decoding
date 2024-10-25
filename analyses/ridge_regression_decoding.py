import argparse
import time

import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import os
from glob import glob
import pickle
from tqdm import trange, tqdm

from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from preprocessing.transform_to_surface import DEFAULT_RESOLUTION
from utils import IMAGERY_SCENES, FMRI_BETAS_DIR, model_features_file_path, VISION_MEAN_FEAT_KEY, \
    VISION_CLS_FEAT_KEY, FUSED_CLS_FEAT_KEY, FUSED_MEAN_FEAT_KEY, LANG_MEAN_FEAT_KEY, \
    LANG_CLS_FEAT_KEY, FMRI_SURFACE_LEVEL_DIR, HEMIS, SUBJECTS, FS_HEMI_NAMES, ACC_CAPTIONS, ACC_IMAGES, \
    ACC_CROSS_CAPTIONS_TO_IMAGES, ACC_CROSS_IMAGES_TO_CAPTIONS, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, \
    ACC_MODALITY_AGNOSTIC

AVG_FEATS = 'avg'
LANG_FEATS_ONLY = 'lang'
VISION_FEATS_ONLY = 'vision'
FUSED_FEATS_CLS = 'fused_cls'
FUSED_FEATS_MEAN = 'fused_mean'
MATCHED_FEATS = 'matched'
FEATS_SELECT_DEFAULT = 'default'
FEATURE_COMBINATION_CHOICES = [AVG_FEATS, LANG_FEATS_ONLY, VISION_FEATS_ONLY, FUSED_FEATS_CLS,
                               FUSED_FEATS_MEAN,
                               MATCHED_FEATS, FEATS_SELECT_DEFAULT]

VISION_FEAT_COMBINATION_CHOICES = [VISION_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY, FEATS_SELECT_DEFAULT]
LANG_FEAT_COMBINATION_CHOICES = [LANG_MEAN_FEAT_KEY, LANG_CLS_FEAT_KEY, FEATS_SELECT_DEFAULT]

NUM_CV_SPLITS = 5
DEFAULT_N_JOBS = 5
DEFAULT_N_PRE_DISPATCH = 5

MOD_SPECIFIC_IMAGES = "train_image"
MOD_SPECIFIC_CAPTIONS = "train_caption"
MODE_AGNOSTIC = "train"
TRAIN_MODE_CHOICES = [MODE_AGNOSTIC, MOD_SPECIFIC_CAPTIONS, MOD_SPECIFIC_IMAGES]
TESTING_MODE = "test"

IMAGE = "image"
CAPTION = "caption"
IMAGERY = "imagery"

DECODER_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/glm/")


def get_default_features(model_name, logging=True):
    if (model_name.startswith("clip") or model_name.startswith("imagebind") or model_name.startswith(
            "flava") or model_name.startswith("random-flava") or model_name.startswith("glow") or model_name.startswith(
        "blip")
    ):
        features = AVG_FEATS
    elif (model_name.startswith("visualbert") or model_name.startswith("lxmert") or model_name.startswith(
            "vilt")):
        features = FUSED_FEATS_MEAN
    elif model_name.startswith("bridgetower"):
        features = FUSED_FEATS_CLS
    elif model_name.startswith("bert") or model_name.startswith("gpt") or model_name.startswith(
            "llama") or model_name.startswith("mistral") or model_name.startswith("mixtral") or model_name.startswith(
        "bge"):
        features = LANG_FEATS_ONLY
    elif model_name.startswith("resnet") or model_name.startswith("vit") or model_name.startswith("dino"):
        features = VISION_FEATS_ONLY
    else:
        raise RuntimeError(f"Unknown default features for {model_name}")

    if logging:
        print(f"Selected default features for {model_name}: {features}")
    return features


def get_default_vision_features(model_name, logging=True):
    vision_feats = VISION_MEAN_FEAT_KEY
    if model_name.startswith("imagebind") or model_name.startswith("clip"):
        vision_feats = VISION_CLS_FEAT_KEY
    elif model_name.startswith("flava") or model_name.startswith("random-flava") or model_name.startswith(
            "blip") or model_name.startswith("resnet-50-glow"):
        vision_feats = VISION_MEAN_FEAT_KEY
    elif model_name.startswith("bridgetower") or model_name.startswith("vilt") or model_name.startswith(
            "visualbert") or model_name.startswith("lxmert") or model_name.startswith("bge"):
        vision_feats = "n_a"
    elif model_name.startswith("bert") or model_name.startswith("llama") or model_name.startswith(
            "mistral") or model_name.startswith("mixtral") or model_name.startswith("gpt"):
        vision_feats = "n_a"

    if logging:
        print(f"Selected default vision features for {model_name}: {vision_feats}")
    return vision_feats


def get_default_lang_features(model_name, logging=True):
    lang_feats = LANG_MEAN_FEAT_KEY
    if model_name.startswith("imagebind") or model_name.startswith("bge") or model_name.startswith(
            "resnet-and-bge") or model_name.startswith("clip"):
        lang_feats = LANG_CLS_FEAT_KEY
    elif model_name.startswith("flava") or model_name.startswith("random-flava") or model_name.startswith("blip"):
        lang_feats = LANG_MEAN_FEAT_KEY
    elif model_name.startswith("bridgetower") or model_name.startswith("vilt") or model_name.startswith(
            "visualbert") or model_name.startswith("lxmert"):
        lang_feats = "n_a"
    elif model_name.startswith("vit") or model_name.startswith("resnet") or model_name.startswith("dino"):
        lang_feats = "n_a"

    if logging:
        print(f"Selected default lang features for {model_name}: {lang_feats}")
    return lang_feats


def get_vision_feats(latent_vectors, stim_id, vision_features_mode):
    if vision_features_mode == VISION_MEAN_FEAT_KEY:
        vision_feats = latent_vectors[stim_id][VISION_MEAN_FEAT_KEY]
    elif vision_features_mode == VISION_CLS_FEAT_KEY:
        vision_feats = latent_vectors[stim_id][VISION_CLS_FEAT_KEY]
    else:
        raise RuntimeError("Unknown vision feature choice: ", vision_features_mode)
    return vision_feats


def get_lang_feats(latent_vectors, stim_id, lang_features_mode):
    if lang_features_mode == LANG_MEAN_FEAT_KEY:
        lang_feats = latent_vectors[stim_id][LANG_MEAN_FEAT_KEY]
    elif lang_features_mode == LANG_CLS_FEAT_KEY:
        lang_feats = latent_vectors[stim_id][LANG_CLS_FEAT_KEY]
    else:
        raise RuntimeError("Unknown lang feature choice: ", lang_features_mode)
    return lang_feats


def get_nn_latent_data(model_name, features, vision_features_mode, lang_features_mode, stim_ids, stim_types, subject,
                       mode,
                       nn_latent_transform=None
                       ):
    latent_vectors_file = model_features_file_path(model_name)
    latent_vectors = pickle.load(open(latent_vectors_file, 'rb'))

    if mode.endswith("_captions"):
        stim_ids = stim_ids[stim_types == CAPTION]
        stim_types = stim_types[stim_types == CAPTION]
    elif mode.endswith("_images"):
        stim_ids = stim_ids[stim_types == IMAGE]
        stim_types = stim_types[stim_types == IMAGE]

    nn_latent_vectors = []
    for stim_id, stim_type in zip(stim_ids, stim_types):
        if features == VISION_FEATS_ONLY:
            feats = get_vision_feats(latent_vectors, stim_id, vision_features_mode)
        elif features == LANG_FEATS_ONLY:
            feats = get_lang_feats(latent_vectors, stim_id, lang_features_mode)
        elif features == AVG_FEATS:
            vision_feats = get_vision_feats(latent_vectors, stim_id, vision_features_mode)
            lang_feats = get_lang_feats(latent_vectors, stim_id, lang_features_mode)
            feats = np.stack((lang_feats, vision_feats))
            feats = feats.mean(axis=0)
        elif features == FUSED_FEATS_CLS:
            feats = latent_vectors[stim_id][FUSED_CLS_FEAT_KEY]
        elif features == FUSED_FEATS_MEAN:
            feats = latent_vectors[stim_id][FUSED_MEAN_FEAT_KEY]
        elif features == MATCHED_FEATS:
            if stim_type == CAPTION:
                feats = get_lang_feats(latent_vectors, stim_id, lang_features_mode)
            elif stim_type == IMAGE:
                feats = get_vision_feats(latent_vectors, stim_id, vision_features_mode)
            elif stim_type == IMAGERY:
                feats = get_vision_feats(latent_vectors, stim_id, vision_features_mode)
            else:
                raise RuntimeError(f"Unknown stim type: {stim_type}")
        else:
            raise RuntimeError(f"Unknown feature selection/combination method: {features}")
        nn_latent_vectors.append(feats)
    nn_latent_vectors = np.array(nn_latent_vectors, dtype=np.float32)

    if nn_latent_transform is None:
        model_std_mean_path = get_latents_mean_std_path(subject, model_name, features, vision_features_mode,
                                                        lang_features_mode, mode)
        os.makedirs(os.path.dirname(model_std_mean_path), exist_ok=True)
        mean_std = dict()
        if mode in [MODE_AGNOSTIC, TESTING_MODE]:
            mean_std[CAPTION] = {
                'mean': nn_latent_vectors[stim_types == CAPTION].mean(axis=0),
                'std': nn_latent_vectors[stim_types == CAPTION].std(axis=0),
            }
            mean_std[IMAGE] = {
                'mean': nn_latent_vectors[stim_types == IMAGE].mean(axis=0),
                'std': nn_latent_vectors[stim_types == IMAGE].std(axis=0),
            }
        else:
            mean_std[CAPTION] = {
                'mean': nn_latent_vectors.mean(axis=0),
                'std': nn_latent_vectors.std(axis=0),
            }
            mean_std[IMAGE] = {
                'mean': nn_latent_vectors.mean(axis=0),
                'std': nn_latent_vectors.std(axis=0),
            }
        pickle.dump(mean_std, open(model_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        nn_latent_transform = load_latents_transform(
            subject, model_name, features, vision_features_mode, lang_features_mode, mode
        )

    nn_latent_vectors = apply_latent_transform(nn_latent_vectors, nn_latent_transform, stim_types)

    return nn_latent_vectors, nn_latent_transform


def apply_latent_transform(nn_latent_vectors, latent_transform, stim_types):
    return np.array([
        latent_transform[CAPTION](v) if type == CAPTION else latent_transform[IMAGE](v)
        for v, type in zip(nn_latent_vectors, stim_types)]
    )


def get_latents_mean_std_path(subject, model_name, features, vision_features_mode, lang_features_mode, mode):
    mean_std_dir = os.path.join(DECODER_OUT_DIR, "normalizations", subject)
    model_std_mean_name = f'{model_name}_{features}_{vision_features_mode}_{lang_features_mode}_mean_std_{mode}.p'
    return os.path.join(mean_std_dir, model_std_mean_name)


def load_latents_transform(subject, model_name, features, vision_features_mode, lang_features_mode, mode):
    model_std_mean_path = get_latents_mean_std_path(
        subject, model_name, features, vision_features_mode, lang_features_mode, mode
    )
    model_mean_std = pickle.load(open(model_std_mean_path, 'rb'))
    nn_latent_transform = {
        CAPTION: Normalize(model_mean_std[CAPTION]['mean'], model_mean_std[CAPTION]['std']),
        IMAGE: Normalize(model_mean_std[IMAGE]['mean'], model_mean_std[IMAGE]['std']),
    }

    return nn_latent_transform


def get_fmri_data_paths(subject, mode, surface=False, hemi=None, resolution=None):
    if surface:
        if hemi is None or resolution is None:
            raise ValueError("Hemi and resolution needs to be specified to load surface-level data")
        filename_suffix = f"*_{hemi}.gii"
        fmri_addresses_regex = os.path.join(
            FMRI_SURFACE_LEVEL_DIR, resolution, subject, f'betas_{mode}*', filename_suffix
        )
    else:
        filename_suffix = "*.nii"
        fmri_addresses_regex = os.path.join(FMRI_BETAS_DIR, subject, f'betas_{mode}*', filename_suffix)

    fmri_betas_paths = sorted(glob(fmri_addresses_regex))
    stim_ids = []
    stim_types = []
    for path in fmri_betas_paths:
        split_name = path.split(os.sep)[-2]
        file_name = os.path.basename(path)
        stim_id = int(file_name.replace('beta_', '').replace(filename_suffix[1:], ''))
        if 'imagery' in split_name:
            stim_types.append(IMAGERY)
            stim_id = IMAGERY_SCENES[subject][stim_id - 1][1]
        elif 'image' in split_name:
            stim_types.append(IMAGE)
        elif 'caption' in split_name:
            stim_types.append(CAPTION)
        else:
            raise RuntimeError(f"Unknown split name: {split_name}")

        stim_ids.append(stim_id)

    stim_ids = np.array(stim_ids)
    stim_types = np.array(stim_types)

    return fmri_betas_paths, stim_ids, stim_types


def get_fmri_voxel_data(subject, mode):
    fmri_betas_paths, stim_ids, stim_types = get_fmri_data_paths(subject, mode)

    gray_matter_mask_path = get_graymatter_mask_path(subject)
    gray_matter_mask_img = nib.load(gray_matter_mask_path)
    gray_matter_mask_data = gray_matter_mask_img.get_fdata()
    gray_matter_mask = gray_matter_mask_data == 1
    print(f"Gray matter mask size: {gray_matter_mask.sum()}")

    fmri_betas = []
    for idx in trange(len(fmri_betas_paths), desc="loading fmri data"):
        sample = nib.load(fmri_betas_paths[idx])
        sample = sample.get_fdata()
        sample = sample[gray_matter_mask].astype('float32').reshape(-1)
        fmri_betas.append(sample)

    fmri_betas = np.array(fmri_betas)
    return fmri_betas, stim_ids, stim_types


def get_fmri_betas_mean_std_path(subject, mode, mask_name):
    mean_std_dir = os.path.join(DECODER_OUT_DIR, "normalizations", subject)
    bold_std_mean_name = f'bold_multimodal_mean_std_{mode}.p'
    if mask_name is not None:
        bold_std_mean_name += f'_mask_{mask_name}'
    return os.path.join(mean_std_dir, bold_std_mean_name)


def load_fmri_betas_transform(subject, mode, mask_name=None):
    bold_std_mean_path = get_fmri_betas_mean_std_path(subject, mode, mask_name)

    bold_mean_std = pickle.load(open(bold_std_mean_path, 'rb'))
    fmri_betas_transform = Normalize(bold_mean_std['mean'], bold_mean_std['std'])
    return fmri_betas_transform


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


def get_distance_matrix(predictions, originals, metric='cosine'):
    return cdist(predictions, originals, metric=metric)


def dist_mat_to_pairwise_acc(dist_mat):
    diag = dist_mat.diagonal().reshape(-1, 1)
    comp_mat = diag < dist_mat
    corrects = comp_mat.sum()
    # subtract the number of elements of the diagonal as these values are always "False" (not smaller than themselves)
    score = corrects / (dist_mat.size - diag.size)
    return score


def pairwise_accuracy(latents, predictions, metric="cosine", normalize_predictions=True, normalize_targets=False):
    if normalize_predictions:
        preds_normalize = Normalize(predictions.mean(axis=0), predictions.std(axis=0))
        predictions = preds_normalize(predictions)
    if normalize_targets:
        latens_normalize = Normalize(latents.mean(axis=0), latents.std(axis=0))
        latents = latens_normalize(latents)

    dist_mat = get_distance_matrix(predictions, latents, metric)
    return dist_mat_to_pairwise_acc(dist_mat)


def pairwise_accuracy_mod_agnostic(latents, predictions, stim_types, metric="cosine", normalize_predictions=True,
                                   normalize_targets=False):
    results = dict()

    if normalize_predictions:
        pred_normalize = Normalize(predictions.mean(axis=0), predictions.std(axis=0))
        predictions = pred_normalize(predictions)
    if normalize_targets:
        latens_normalize = Normalize(latents.mean(axis=0), latents.std(axis=0))
        latents = latens_normalize(latents)

    dist_mat = get_distance_matrix(predictions, latents, metric)

    for modality in [CAPTION, IMAGE]:
        dist_mat_within_mod = dist_mat[stim_types == modality][:, stim_types == modality]
        dist_mat_cross_modal = dist_mat[stim_types == modality][:, stim_types != modality]
        dist_mat_min = np.min((dist_mat_within_mod, dist_mat_cross_modal), axis=0)
        results[f"pairwise_acc_mod_agnostic_{modality}s"] = dist_mat_to_pairwise_acc(dist_mat_min)

    score = np.mean([results[f"pairwise_acc_mod_agnostic_{modality}s"] for modality in [CAPTION, IMAGE]])

    results[ACC_MODALITY_AGNOSTIC] = score
    return results


def calc_all_pairwise_accuracy_scores(latents, predictions, stim_types=None, imagery_latents=None,
                                      imagery_predictions=None, metric="cosine", normalize_predictions=True,
                                      normalize_targets=False, norm_imagery_preds_with_test_preds=False,
                                      comp_mod_agnostic_scores=True, comp_cross_decoding_scores=True):
    results = dict()
    if comp_mod_agnostic_scores:
        results.update(
            pairwise_accuracy_mod_agnostic(
                latents, predictions, stim_types, metric, normalize_predictions,
                normalize_targets
            )
        )

    for modality, acc_metric_name in zip([CAPTION, IMAGE], [ACC_CAPTIONS, ACC_IMAGES]):
        preds_mod = predictions[stim_types == modality].copy()
        latents_mod = latents[stim_types == modality]

        results[acc_metric_name] = pairwise_accuracy(latents_mod, preds_mod, metric, normalize_predictions,
                                                     normalize_targets)

    if comp_cross_decoding_scores:
        for mod_preds, mod_latents, acc_metric_name in zip([CAPTION, IMAGE], [IMAGE, CAPTION],
                                                           [ACC_CROSS_CAPTIONS_TO_IMAGES,
                                                            ACC_CROSS_IMAGES_TO_CAPTIONS]):
            preds_mod = predictions[stim_types == mod_preds].copy()
            latents_mod = latents[stim_types == mod_latents]

            results[acc_metric_name] = pairwise_accuracy(latents_mod, preds_mod, metric, normalize_predictions,
                                                         normalize_targets)

    if imagery_latents is not None:
        results.update(
            calc_imagery_pairwise_accuracy_scores(
                imagery_latents, imagery_predictions, latents, metric,
                normalize_predictions, normalize_targets,
                test_set_preds=predictions if norm_imagery_preds_with_test_preds else None
            )
        )

    return results


def calc_imagery_pairwise_accuracy_scores(imagery_latents, imagery_predictions, latents, metric="cosine",
                                          normalize_predictions=True, normalize_targets=False, test_set_preds=None):
    results = dict()

    results[ACC_IMAGERY] = pairwise_accuracy(
        imagery_latents, imagery_predictions, metric, normalize_predictions, normalize_targets
    )

    if test_set_preds is not None:
        all_preds = np.concatenate((imagery_predictions, test_set_preds))
        norm = Normalize(all_preds.mean(axis=0), all_preds.std(axis=0))
        imagery_predictions = norm(imagery_predictions)

        normalize_predictions = False  # Do not normalize again

    target_latents = np.concatenate((imagery_latents, latents))
    results[ACC_IMAGERY_WHOLE_TEST] = pairwise_accuracy(
        target_latents, imagery_predictions, metric, normalize_predictions, normalize_targets
    )

    return results


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
    latent_1_images = latent_1[stimulus_types == IMAGE]
    latent_2_images = latent_2[stimulus_types == IMAGE]
    return calc_rsa(latent_1_images, latent_2_images, metric, matrix_metric)


def calc_rsa_captions(latent_1, latent_2, stimulus_types, metric="spearmanr", matrix_metric="spearmanr"):
    assert len(latent_1) == len(latent_2) == len(stimulus_types)
    latent_1_captions = latent_1[stimulus_types == CAPTION]
    latent_2_captions = latent_2[stimulus_types == CAPTION]
    return calc_rsa(latent_1_captions, latent_2_captions, metric, matrix_metric)


def get_run_str(model_name, features, test_features, vision_features, lang_features, mask, surface, resolution):
    run_str = f"{model_name}_{features}_test_{test_features}"
    run_str += f"_{vision_features}"
    run_str += f"_{lang_features}"

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

    return run_str


def get_fmri_surface_data(subject, mode, resolution):
    base_mode = mode.split('_')[0]
    print(f"loading {base_mode} fmri data.. ", end="")
    fmri_betas = {
        hemi: pickle.load(
            open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_{hemi}_{resolution}_{base_mode}.p"), 'rb')) for hemi
        in HEMIS
    }
    print("done.")
    stim_ids = pickle.load(open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_stim_ids_{base_mode}.p"), 'rb'))
    stim_types = pickle.load(open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_stim_types_{base_mode}.p"), 'rb'))

    if mode == MOD_SPECIFIC_CAPTIONS:
        for hemi in HEMIS:
            fmri_betas[hemi] = fmri_betas[hemi][stim_types == CAPTION]
        stim_ids = stim_ids[stim_types == CAPTION]
        stim_types = stim_types[stim_types == CAPTION]
    elif mode == MOD_SPECIFIC_IMAGES:
        for hemi in HEMIS:
            fmri_betas[hemi] = fmri_betas[hemi][stim_types == IMAGE]
        stim_ids = stim_ids[stim_types == IMAGE]
        stim_types = stim_types[stim_types == IMAGE]

    return fmri_betas, stim_ids, stim_types


def get_fmri_data(subject, mode, surface=False, resolution=None):
    if surface:
        fmri_betas, stim_ids, stim_types = get_fmri_surface_data(subject, mode, resolution)
        fmri_betas = np.concatenate((fmri_betas[HEMIS[0]], fmri_betas[HEMIS[1]]), axis=1)
    else:
        fmri_betas, stim_ids, stim_types = get_fmri_voxel_data(subject, mode)

    return fmri_betas, stim_ids, stim_types


def load_surface_mask(mask_path):
    mask = pickle.load(open(mask_path, 'rb'))
    return mask


def apply_mask_and_clean(mask_name, betas_list, args):
    if mask_name is not None:
        if not args.surface:
            raise NotImplementedError()
        mask = load_surface_mask(mask_name)
        mask_flat = np.concatenate((mask[HEMIS[0]], mask[HEMIS[1]]))
        betas_list = [betas[:, mask_flat == 1].copy() for betas in betas_list]

    betas_list = [betas[:, ~np.isnan(betas[0])] for betas in betas_list]

    return betas_list


def normalize_fmri_betas(train_fmri_betas, test_fmri_betas, imagery_fmri_betas, subject, training_mode, mask_name):
    bold_std_mean_path = get_fmri_betas_mean_std_path(subject, training_mode, mask_name)
    os.makedirs(os.path.dirname(bold_std_mean_path), exist_ok=True)

    mean_std = {'mean': train_fmri_betas.mean(axis=0),
                'std': train_fmri_betas.std(axis=0)}
    pickle.dump(mean_std, open(bold_std_mean_path, 'wb'))

    fmri_betas_transform = load_fmri_betas_transform(subject, training_mode, mask_name)

    train_fmri_betas = np.apply_along_axis(func1d=fmri_betas_transform, axis=1, arr=train_fmri_betas)
    test_fmri_betas = np.apply_along_axis(func1d=fmri_betas_transform, axis=1, arr=test_fmri_betas)
    imagery_fmri_betas = np.apply_along_axis(func1d=fmri_betas_transform, axis=1, arr=imagery_fmri_betas)

    return train_fmri_betas, test_fmri_betas, imagery_fmri_betas


def run(args):
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
                train_fmri_betas, test_fmri_betas, imagery_fmri_betas = normalize_fmri_betas(
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

                                train_latents, latent_transform = get_nn_latent_data(
                                    model_name, features,
                                    vision_features,
                                    lang_features,
                                    train_stim_ids,
                                    train_stim_types,
                                    subject,
                                    training_mode,
                                )

                                model = Ridge()
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
                                    "mask": mask,
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

                                results_dir = os.path.join(DECODER_OUT_DIR, training_mode, subject)
                                run_str = get_run_str(
                                    model_name, features, test_features, vision_features, lang_features, mask,
                                    args.surface,
                                    args.resolution)
                                results_file_dir = os.path.join(results_dir, run_str)
                                os.makedirs(results_file_dir, exist_ok=True)

                                pickle.dump(results, open(os.path.join(results_file_dir, "results.p"), 'wb'))


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
                        default=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7])

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-pre-dispatch-jobs", type=int, default=DEFAULT_N_PRE_DISPATCH)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(DECODER_OUT_DIR, exist_ok=True)

    run(args)
