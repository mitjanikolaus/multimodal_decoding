import argparse
import time

import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_destrieux_2009
from nilearn.image import resample_to_img
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import os
from glob import glob
import pickle
from tqdm import trange
import pandas as pd

from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, model_features_file_path, VISION_MEAN_FEAT_KEY, \
    VISION_CLS_FEAT_KEY, ROOT_DIR, FUSED_CLS_FEAT_KEY, FUSED_MEAN_FEAT_KEY, LANG_MEAN_FEAT_KEY, \
    LANG_CLS_FEAT_KEY, FMRI_SURFACE_LEVEL_DIR, HEMIS

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

DEFAULT_SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']

MOD_SPECIFIC_IMAGES = "train_images"
MOD_SPECIFIC_CAPTIONS = "train_captions"
MODE_AGNOSTIC = "train"
TRAIN_MODE_CHOICES = [MODE_AGNOSTIC, MOD_SPECIFIC_CAPTIONS, MOD_SPECIFIC_IMAGES]
TESTING_MODE = "test"

ACC_MODALITY_AGNOSTIC = "pairwise_acc_modality_agnostic"
ACC_CAPTIONS = "pairwise_acc_captions"
ACC_IMAGES = "pairwise_acc_images"

IMAGE = "image"
CAPTION = "caption"

DECODER_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/glm/")

MASK_ANATOMICAL_LANGUAGE = "anatomical_lang"
MASK_ANATOMICAL_VISUAL_LOW_LEVEL = "anatomical_visual_low_level"
MASK_ANATOMICAL_VISUAL_HIGH_LEVEL = "anatomical_visual_high_level"

REGIONS_LOW_LEVEL_VISUAL = [
    'L G_and_S_occipital_inf',
    'L G_occipital_middle',
    'L G_occipital_sup',
    'L Pole_occipital',
    'L S_oc_middle_and_Lunatus',
    'L S_oc_sup_and_transversal',
    'L S_occipital_ant',
    'R G_and_S_occipital_inf',
    'R G_occipital_middle',
    'R G_occipital_sup',
    'R Pole_occipital',
    'R S_oc_middle_and_Lunatus',
    'R S_oc_sup_and_transversal',
    'R S_occipital_ant',
    'L S_calcarine',
    'R S_calcarine',
    'L G_cuneus',
    'R G_cuneus',
    'L G_oc-temp_med-Lingual',
    'R G_oc-temp_med-Lingual',
]

REGIONS_HIGH_LEVEL_VISUAL = [
    'L G_oc-temp_lat-fusifor',
    'R G_oc-temp_lat-fusifor',
    'L G_oc-temp_med-Parahip',
    'R G_oc-temp_med-Parahip',
    'L S_oc-temp_lat',
    'R S_oc-temp_lat',
    'L G_temporal_inf',
    'L G_temporal_middle',
    'L S_temporal_inf',
    'R G_temporal_inf',
    'R G_temporal_middle',
    'R S_temporal_inf',
    'L S_collat_transv_post',
    'R S_collat_transv_post',
    'L S_parieto_occipital',
    'R S_parieto_occipital',
    'L S_oc-temp_med_and_Lingual',
    'R S_oc-temp_med_and_Lingual',
]

REGIONS_LANGUAGE = [
    'L G_front_inf-Opercular',  # left inferior frontal gyrus
    'L G_front_inf-Orbital',  # left orbital inferior frontal gyrus
    'L G_front_inf-Triangul',  # left inferior frontal gyrus
    'L G_pariet_inf-Angular',  # left angular gyrus
    'L G_front_middle',  # left middle frontal gyrus
    'L G_temp_sup-Lateral',  # lateral aspect of the superior temporal gyrus: middle-anterior temporal lobe
    'L G_temp_sup-Plan_tempo',  # Planum temporale of the superior temporal gyrus
    'L G_temp_sup-Plan_polar',  # Planum polare of the superior temporal gyrus
    'L G_and_S_subcentral',  # Subcentral gyrus (central operculum) and sulci
    'L S_temporal_sup',  # Superior temporal sulcus
    'L S_temporal_transverse',  # Transverse temporal sulcus
    'L G_temp_sup-G_T_transv',  # Anterior transverse temporal gyrus
    'L G_pariet_inf-Supramar',  # Supramarginal gyrus
    'L G_Ins_lg_and_S_cent_ins',  # Insula
    'L G_insular_short',  # Insula
    'L S_circular_insula_ant',  # Insula
    'L S_circular_insula_inf',  # Insula
    'L S_circular_insula_sup',  # Insula
]


def get_anatomical_mask(roi_mask_name):
    destrieux_atlas = fetch_atlas_destrieux_2009()
    label_to_id_dict = {label[1]: int(label[0]) for label in destrieux_atlas['labels']}
    atlas_map = nib.load(destrieux_atlas.maps).get_fdata()
    if roi_mask_name == MASK_ANATOMICAL_VISUAL_LOW_LEVEL:
        region_names = [label for label in REGIONS_LOW_LEVEL_VISUAL]
    elif roi_mask_name == MASK_ANATOMICAL_VISUAL_HIGH_LEVEL:
        region_names = [label for label in REGIONS_HIGH_LEVEL_VISUAL]
    elif roi_mask_name == MASK_ANATOMICAL_LANGUAGE:
        region_names = [label for label in REGIONS_LANGUAGE]
    else:
        raise RuntimeError("Unknown mask: ", roi_mask_name)

    ids = [label_to_id_dict[label] for label in region_names]
    roi_mask = np.isin(atlas_map, ids)
    return roi_mask


MASK_FUNCTIONAL_LANGUAGE = "functional_Language"
MASK_FUNCTIONAL_VISUAL1 = "functional_Visual1"
MASK_FUNCTIONAL_VISUAL2 = "functional_Visual2"


def get_functional_mask(roi_mask_name, ref_img):
    ji_conv_filename = os.path.join(ROOT_DIR,
                                    'atlas_data/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt')
    ji_conversion = pd.read_csv(ji_conv_filename, delimiter='\t')

    network_names = roi_mask_name.split("_")[1:]
    glasser_labels = ji_conversion[ji_conversion.NETWORK.isin(network_names)].GLASSERLABELNAME.dropna().unique()

    atlas_hcp = nib.load(os.path.join(ROOT_DIR, 'atlas_data/MNI_Glasser_HCP_v1.0.nii.gz'))
    hcp_resampled = resample_to_img(atlas_hcp, ref_img, interpolation='nearest')
    hcp_data = hcp_resampled.get_fdata().round().astype(np.int32)

    glasser_label_to_idx = pd.read_csv(os.path.join(ROOT_DIR, 'atlas_data/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt'),
                                       delimiter=' ', names=['idx', 'label'], index_col=1)

    def get_idx(label):
        idx = glasser_label_to_idx.loc[label.replace('R_', 'L_')].idx
        if label.startswith("R_"):
            idx += 1000
        return idx

    ids = np.unique([get_idx(label) for label in glasser_labels])
    roi_mask = np.isin(hcp_data, ids)
    return roi_mask


def get_roi_mask(roi_mask_name, ref_img):
    if roi_mask_name.startswith("anatomical_"):
        return get_anatomical_mask(roi_mask_name)

    elif roi_mask_name.startswith("functional_"):
        return get_functional_mask(roi_mask_name, ref_img)

    else:
        raise RuntimeError("Unknown mask for volume space: ", roi_mask_name)


def get_default_features(model_name):
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

    print(f"Selected default features for {model_name}: {features}")
    return features


def get_default_vision_features(model_name):
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

    print(f"Selected default vision features for {model_name}: {vision_feats}")
    return vision_feats


def get_default_lang_features(model_name):
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
                       nn_latent_transform=None,
                       recompute_std_mean=False):
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
            else:
                raise RuntimeError(f"Unknown stim type: {stim_type}")
        else:
            raise RuntimeError(f"Unknown feature selection/combination method: {features}")
        nn_latent_vectors.append(feats)
    nn_latent_vectors = np.array(nn_latent_vectors, dtype=np.float32)

    if nn_latent_transform is None:
        model_std_mean_path = get_latents_mean_std_path(subject, model_name, features, vision_features_mode,
                                                        lang_features_mode, mode)
        if not os.path.exists(model_std_mean_path) or recompute_std_mean:
            print(f"Calculating Mean and STD of Model Latent Variables for {mode} samples")
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


def get_fmri_voxel_data(subject, mode, fmri_betas_transform=None, roi_mask_name=None, recompute_std_mean=False):
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
            stim_types.append(IMAGE)
        elif 'C' in file_name:  # Caption
            stim_id = int(file_name[file_name.find('C') + 1:-4])
            stim_types.append(CAPTION)
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
    print(f"Gray matter mask size: {gray_matter_mask.sum()}")

    if roi_mask_name is not None:
        roi_mask = get_roi_mask(roi_mask_name, gray_matter_mask_ras)
        print(f"Applying ROI {roi_mask_name} mask of size {roi_mask.sum()}")
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
        bold_std_mean_path = get_fmri_betas_mean_std_path(subject, mode, roi_mask_name)

        if not os.path.exists(bold_std_mean_path) or recompute_std_mean:
            print(f"Calculating mean and std of BOLD Signals for mode {mode} with mask {roi_mask_name}")
            os.makedirs(os.path.dirname(bold_std_mean_path), exist_ok=True)

            mean_std = {'mean': fmri_betas.mean(axis=0),
                        'std': fmri_betas.std(axis=0)}
            pickle.dump(mean_std, open(bold_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        fmri_betas_transform = load_fmri_betas_transform(subject, mode, roi_mask_name)

    fmri_betas = np.array([fmri_betas_transform(v) for v in fmri_betas])

    return fmri_betas, stim_ids, stim_types, fmri_betas_transform


def get_fmri_betas_mean_std_path(subject, mode, roi_mask_name):
    mean_std_dir = os.path.join(DECODER_OUT_DIR, "normalizations", subject)
    bold_std_mean_name = f'bold_multimodal_mean_std_{mode}.p'
    if roi_mask_name is not None:
        bold_std_mean_name += f'_mask_{roi_mask_name}'
    return os.path.join(mean_std_dir, bold_std_mean_name)


def load_fmri_betas_transform(subject, mode, roi_mask_name=None):
    bold_std_mean_path = get_fmri_betas_mean_std_path(subject, mode, roi_mask_name)

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


def all_pairwise_accuracy_scores(latents, predictions, stim_types=None, metric="cosine", normalize=True):
    results = dict()

    for modality, acc_metric_name in zip([CAPTION, IMAGE], [ACC_CAPTIONS, ACC_IMAGES]):
        preds_mod = predictions[stim_types == modality].copy()
        latents_mod = latents[stim_types == modality]
        if normalize:
            pred_mod_normalize = Normalize(preds_mod.mean(axis=0), preds_mod.std(axis=0))
            preds_mod = pred_mod_normalize(preds_mod)

        dist_mat = get_distance_matrix(preds_mod, latents_mod, metric)
        diag = dist_mat.diagonal().reshape(-1, 1)
        comp_mat = diag < dist_mat
        results[acc_metric_name] = comp_mat.mean()

    if normalize:
        pred_normalize = Normalize(predictions.mean(axis=0), predictions.std(axis=0))
        predictions = pred_normalize(predictions)

    dist_mat = get_distance_matrix(predictions, latents, metric)

    mod_agnostic_accs = []
    for modality in [CAPTION, IMAGE]:
        dist_mat_within_mod = dist_mat[stim_types == modality][:, stim_types == modality]
        dist_mat_cross_modal = dist_mat[stim_types == modality][:, stim_types != modality]
        dist_mat_min = np.min((dist_mat_within_mod, dist_mat_cross_modal), axis=0)
        diag = dist_mat_min.diagonal().reshape(-1, 1)
        comp_mat = diag < dist_mat_min
        scores = np.mean(comp_mat, axis=0)
        mod_agnostic_accs.extend(scores)
        results[f"pairwise_acc_mod_agnostic_{modality}s"] = scores.mean()

    results[ACC_MODALITY_AGNOSTIC] = np.mean(mod_agnostic_accs)

    return results


def pairwise_accuracy(latents, predictions, metric="cosine", normalize=True):
    if normalize:
        pred_normalize = Normalize(predictions.mean(axis=0), predictions.std(axis=0))
        predictions = pred_normalize(predictions)

    if "csls_" in metric:
        metric = metric.replace("csls_", "")
        dist_mat = get_distance_matrix_csls(predictions, latents, metric=metric)
    else:
        dist_mat = get_distance_matrix(predictions, latents, metric)

    diag = dist_mat.diagonal().reshape(-1, 1)  # all congruent distances
    comp_mat = diag < dist_mat  # we are interested in i,j where d(i,i) < d(i,j)

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
    latent_1_images = latent_1[stimulus_types == IMAGE]
    latent_2_images = latent_2[stimulus_types == IMAGE]
    return calc_rsa(latent_1_images, latent_2_images, metric, matrix_metric)


def calc_rsa_captions(latent_1, latent_2, stimulus_types, metric="spearmanr", matrix_metric="spearmanr"):
    assert len(latent_1) == len(latent_2) == len(stimulus_types)
    latent_1_captions = latent_1[stimulus_types == CAPTION]
    latent_2_captions = latent_2[stimulus_types == CAPTION]
    return calc_rsa(latent_1_captions, latent_2_captions, metric, matrix_metric)


def get_run_str(model_name, features, vision_features, lang_features, mask, surface, resolution):
    run_str = f"{model_name}_{features}"
    run_str += f"_{vision_features}"
    run_str += f"_{lang_features}"

    if mask is not None:
        if mask.startswith("functional_") or mask.startswith("anatomical_"):
            run_str += f"_mask_{mask}"
        elif "p_values" in mask:
            run_str += f"_mask_p_values"
        else:
            raise RuntimeError(f"Unsupported mask: {mask}")

    if surface:
        run_str += f"_surface_{resolution}"

    return run_str


def get_fmri_surface_data(subject, mode, resolution):
    base_mode = mode.split('_')[0]
    fmri_betas = {
        hemi: pickle.load(
            open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_{hemi}_{resolution}_{base_mode}.p"), 'rb')) for hemi
        in HEMIS
    }
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


def get_surface_mask(mask_path, p_value_mask_threshold):
    masks = dict()
    if "p_values" in mask_path:
        print(f"loading surface space mask based on p values from {mask_path}")
        print(f"p value threshold: {p_value_mask_threshold}")
        p_values = pickle.load(open(mask_path, 'rb'))
        for hemi in HEMIS:
            masks[hemi] = p_values[hemi] < p_value_mask_threshold
            masks[hemi][p_values[hemi] == 0] = False
    else:
        raise RuntimeError(f"Unsupported mask type for surface-based fmri: {mask_path}")
    return masks


def get_fmri_data(subject, mode, mask_name=None, fmri_transform=None, recompute_std_mean=False, surface=False,
                  resolution=None, p_value_mask_threshold=0.01):
    if surface:
        fmri_betas, stim_ids, stim_types = get_fmri_surface_data(subject, mode, resolution)

        if mask_name is not None:
            mask = get_surface_mask(mask_name, p_value_mask_threshold)
            for hemi in HEMIS:
                fmri_betas[hemi] = fmri_betas[hemi][:, mask[hemi]]

        fmri_betas = np.concatenate((fmri_betas['left'], fmri_betas['right']), axis=1)

        if fmri_transform is None:
            fmri_transform = Normalize(fmri_betas.mean(axis=0), fmri_betas.std(axis=0))
        fmri_betas = np.array([fmri_transform(v) for v in fmri_betas])

        fmri_betas = fmri_betas[:, ~np.isnan(fmri_betas[0])]

    else:
        fmri_betas, stim_ids, stim_types, fmri_transform = get_fmri_voxel_data(
            subject,
            mode,
            roi_mask_name=mask_name,
            recompute_std_mean=recompute_std_mean,
            fmri_betas_transform=fmri_transform,
        )

    print(f"fMRI betas shape: {fmri_betas.shape}")
    return fmri_betas, stim_ids, stim_types, fmri_transform


def run(args):
    for training_mode in args.training_modes:
        for mask in args.masks:
            mask = None if mask in ["none", "None"] else mask
            for subject in args.subjects:
                train_fmri_betas, train_stim_ids, train_stim_types, fmri_transform = get_fmri_data(
                    subject,
                    training_mode,
                    mask_name=mask,
                    recompute_std_mean=args.recompute_std_mean,
                    surface=args.surface,
                    resolution=args.resolution,
                    p_value_mask_threshold=args.p_value_mask_threshold,
                )
                test_fmri_betas, test_stim_ids, test_stim_types, _ = get_fmri_data(
                    subject,
                    TESTING_MODE,
                    fmri_transform=fmri_transform,
                    mask_name=mask,
                    surface=args.surface,
                    resolution=args.resolution,
                    p_value_mask_threshold=args.p_value_mask_threshold,
                )

                for model_name in args.models:
                    model_name = model_name.lower()

                    for features in args.features:
                        if features == FEATS_SELECT_DEFAULT:
                            features = get_default_features(model_name)
                        for vision_features in args.vision_features:
                            if vision_features == FEATS_SELECT_DEFAULT:
                                vision_features = get_default_vision_features(model_name)
                            for lang_features in args.lang_features:
                                if lang_features == FEATS_SELECT_DEFAULT:
                                    lang_features = get_default_lang_features(model_name)

                                print(f"\nTRAIN MODE: {training_mode} | MASK: {mask} | SUBJECT: {subject} | "
                                      f"MODEL: {model_name} | FEATURES: {features} {vision_features} {lang_features}")

                                train_latents, latent_transform = get_nn_latent_data(
                                    model_name, features,
                                    vision_features,
                                    lang_features,
                                    train_stim_ids,
                                    train_stim_types,
                                    subject,
                                    training_mode,
                                    recompute_std_mean=args.recompute_std_mean
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

                                test_data_latents, _ = get_nn_latent_data(model_name, features, vision_features,
                                                                          lang_features,
                                                                          test_stim_ids,
                                                                          test_stim_types,
                                                                          subject,
                                                                          TESTING_MODE,
                                                                          nn_latent_transform=latent_transform)
                                best_model = clf.best_estimator_
                                test_predicted_latents = best_model.predict(test_fmri_betas)

                                results = {
                                    "alpha": best_alpha,
                                    "model": model_name,
                                    "subject": subject,
                                    "features": features,
                                    "vision_features": vision_features,
                                    "lang_features": lang_features,
                                    "training_mode": training_mode,
                                    "mask": mask,
                                    "num_voxels": test_fmri_betas.shape[1],
                                    "cv_results": clf.cv_results_,
                                    "stimulus_ids": test_stim_ids,
                                    "stimulus_types": test_stim_types,
                                    "predictions": test_predicted_latents,
                                    "latents": test_data_latents,
                                    "surface": args.surface,
                                    "resolution": args.resolution,
                                    "p_value_mask_threshold": args.p_value_mask_threshold,
                                }
                                results.update(
                                    all_pairwise_accuracy_scores(
                                        test_data_latents, test_predicted_latents, test_stim_types
                                    )
                                )
                                print(f"Best alpha: {best_alpha}"
                                      f" | Pairwise acc (mod-agnostic): {results[ACC_MODALITY_AGNOSTIC]:.2f}"
                                      f" | Pairwise acc (captions): {results[ACC_CAPTIONS]:.2f}"
                                      f" | Pairwise acc (images): {results[ACC_IMAGES]:.2f}")

                                results_dir = os.path.join(DECODER_OUT_DIR, training_mode, subject)
                                run_str = get_run_str(
                                    model_name, features, vision_features, lang_features, mask, args.surface,
                                    args.resolution)
                                results_file_dir = os.path.join(results_dir, run_str)
                                os.makedirs(results_file_dir, exist_ok=True)

                                pickle.dump(results, open(os.path.join(results_file_dir, "results.p"), 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--surface", action="store_true", default=False)
    parser.add_argument("--resolution", type=str, default="fsaverage7")
    parser.add_argument("--p-value-mask-threshold", type=float, default=0.01)

    parser.add_argument("--models", type=str, nargs='+', default=['blip2'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=VISION_FEAT_COMBINATION_CHOICES)
    parser.add_argument("--lang-features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=LANG_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--masks", type=str, nargs='+', default=[None])

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--l2-regularization-alphas", type=float, nargs='+', default=[1e3, 1e5, 1e7])

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--n-pre-dispatch-jobs", type=int, default=DEFAULT_N_PRE_DISPATCH)

    parser.add_argument("--recompute-std-mean", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(DECODER_OUT_DIR, exist_ok=True)

    run(args)
