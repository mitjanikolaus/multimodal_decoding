import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data import DEFAULT_FEATURES, DEFAULT_VISION_FEATURES, DEFAULT_LANG_FEATURES, TRAINING_MODES, MODALITY_AGNOSTIC, MODALITY_SPECIFIC_IMAGES, MODALITY_SPECIFIC_CAPTIONS
from data import MODALITY_AGNOSTIC, MODALITY_SPECIFIC_IMAGES, MODALITY_SPECIFIC_CAPTIONS, TRAINING_MODES, CAPTION, IMAGE, SPLIT_TRAIN, TEST_SPLITS, SPLIT_IMAGERY, SPLIT_IMAGERY_WEAK, SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS, LatentFeatsConfig, get_stim_info, get_latents_for_splits, standardize_latents
from eval import ACC_MODALITY_AGNOSTIC, ACC_CROSS_IMAGES_TO_CAPTIONS, ACC_CROSS_CAPTIONS_TO_IMAGES, \
    calc_all_pairwise_accuracy_scores, ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST
from utils import SUBJECTS, RIDGE_DECODER_OUT_DIR, DECODER_ADDITIONAL_TEST_OUT_DIR
from analyses.decoding.ridge_regression_decoding import get_run_str, RESULTS_FILE, PREDICTIONS_FILE

from tqdm import tqdm
from glob import glob
import pickle
import os

HP_KEYS = ["alpha", "model", "subject", "features", "test_features", "vision_features", "lang_features",
           "training_mode", "mask",
           "num_voxels", "surface", "resolution"]
METRIC_NAMES = {"acc_cosine": "pairwise_acc", "acc_cosine_captions": "pairwise_acc_captions",
                "acc_cosine_images": "pairwise_acc_images"}

ACC_MEAN = "pairwise_acc_mean"
ACC_CROSS_MEAN = "pairwise_acc_cross_mean"
FEATS_MULTIMODAL = ["fused_mean", "fused_cls", "avg", "matched"]
DEFAULT_FEAT_OPTIONS = ["vision", "lang"] + FEATS_MULTIMODAL


def load_predictions(betas_dir, subject, mode, feats_config, surface, mask=None, training_splits=[SPLIT_TRAIN], imagery_samples_weight=None):    
    run_str = get_run_str(betas_dir, feats_config, surface=surface, mask=mask, training_splits=training_splits, imagery_samples_weight=imagery_samples_weight)
    predictions_file_path = os.path.join(
        DECODER_ADDITIONAL_TEST_OUT_DIR, mode, subject, run_str, PREDICTIONS_FILE
    )
    print(f'loading decoder results from: \n', predictions_file_path)
    results = pickle.load(open(predictions_file_path, 'rb'))
    
    return results    


def load_betas(train_paths, test_paths):
    train_fmri_betas = []
    for idx in trange(len(train_paths), desc="loading fmri data"):
        sample = nibabel.load(train_paths[idx]).get_fdata()
        sample = sample[gray_matter_mask].astype('float32').reshape(-1)
        train_fmri_betas.append(sample)
    
    train_fmri_betas = np.array(train_fmri_betas)

    test_fmri_betas = []
    for idx in trange(len(test_paths), desc="loading fmri data"):
        sample = nibabel.load(test_paths[idx]).get_fdata()
        sample = sample[gray_matter_mask].astype('float32').reshape(-1)
        test_fmri_betas.append(sample)
    
    test_fmri_betas = np.array(test_fmri_betas)

    scaler = StandardScaler()
    scaler.fit(train_fmri_betas)
    train_fmri_betas_standardized = scaler.transform(train_fmri_betas)
    test_fmri_betas_standardized = scaler.transform(test_fmri_betas)
    
    return train_fmri_betas, test_fmri_betas, train_fmri_betas_standardized, test_fmri_betas_standardized



def calc_model_feat_order(data, ref_models, feat_options=DEFAULT_FEAT_OPTIONS):
    all_model_feats = data.model_feat.unique()
    all_models = data.model.unique()
    for model in all_models:
        if model not in ref_models:
            raise RuntimeError(f"Model missing in order: {model}")
    model_feat_order = []
    for model in ref_models:
        for feats in feat_options:
            model_feat = f"{model}_{feats}"
            if model_feat in all_model_feats:
                model_feat_order.append(model_feat)

    return model_feat_order
    

def get_data_default_feats(data):
    data_default_feats = data.copy()
    for model in data.model.unique():
        default_feats = DEFAULT_FEATURES[model]
        default_vision_feats = DEFAULT_VISION_FEATURES[model]
        default_lang_feats = DEFAULT_LANG_FEATURES[model]
        data_default_feats = data_default_feats[
            ((data_default_feats.model == model) & (data_default_feats.features == default_feats) & (
                    data_default_feats.vision_features == default_vision_feats) & (
                     data_default_feats.lang_features == default_lang_feats)) | (data_default_feats.model != model)
            ]

    return data_default_feats
