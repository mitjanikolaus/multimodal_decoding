import logging
import os
import pickle
from dataclasses import dataclass
from glob import glob

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
import nibabel as nib
from tqdm import tqdm, trange

from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import model_features_file_path, FMRI_NORMALIZATIONS_DIR, \
    LATENT_FEATURES_NORMALIZATIONS_DIR, FMRI_SURFACE_LEVEL_DIR, MOD_SPECIFIC_CAPTIONS, MOD_SPECIFIC_IMAGES, HEMIS
import lightning as pl

MODALITY_SPECIFIC_IMAGES = "images"
MODALITY_SPECIFIC_CAPTIONS = "captions"
MODALITY_AGNOSTIC = "agnostic"

TRAIN_MODE_CHOICES = [MODALITY_AGNOSTIC, MODALITY_SPECIFIC_CAPTIONS, MODALITY_SPECIFIC_IMAGES]

TESTING_MODE = "test"

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
SPLIT_IMAGERY = "imagery"

IMAGE = "image"
CAPTION = "caption"
IMAGERY = "imagery"

IMAGES_IMAGERY_CONDITION = [
    [406591, f'images/train2017/000000406591.jpg',
     'A woman sits in a beach chair as a man walks along the sand'],
    [324670, f'images/train2017/000000324670.jpg',
     'White bird sitting in front of a lighthouse with a red roof'],
    [563723, f'images/train2017/000000563723.jpg',
     'A little cat standing on the lap of a man sitting in a chair'],
    [254568, f'images/train2017/000000254568.jpg',
     'A lonely giraffe is walking in the middle of a grassy field'],
    [42685, f'images/train2017/000000042685.jpg',
     'A party of young people in a bedroom with a large box of pizza'],
    [473773, f'images/train2017/000000473773.jpg',
     'A man walking across a field of snow holding skis and ski poles'],
    [485909, f'images/train2017/000000485909.jpg',
     'Two men are discussing something next to a desk with a laptop'],
    [544502, f'images/train2017/000000544502.jpg',
     'A young male holding a racket and tennis ball in a tennis court'],
    [556512, f'images/train2017/000000556512.jpg',
     'A group of kids swimming in the ocean around a man on a surfboard'],
    [410573, f'images/train2017/000000410573.jpg',
     'A cat and a dog staring intensely at each other on an armchair'],
    [48670, f'images/train2017/000000048670.jpg',
     'A man stands by a rainy street with an umbrella over his head'],
    [263212, f'images/train2017/000000263212.jpg',
     'A woman working on her computer while also using her cell phone'],
    [214816, f'images/train2017/000000214816.jpg',
     'An old broken down church and graffiti on surrounding structures'],
    [141343, f'images/train2017/000000141343.jpg',
     'A teddy bear lying down on the sidewalk in front of a store'],
    [213506, f'images/train2017/000000213506.jpg',
     'A woman leaning out a window to talk to someone on the sidewal'],
    [162396, f'images/train2017/000000162396.jpg',
     'The man on the skateboard and the dog are getting their picture taken'],
]

IMAGERY_SCENES = {
    'sub-01':
        [
            ('A woman sits in a beach chair as a man walks along the sand', 406591),
            ('White bird sitting in front of a lighthouse with a red roof', 324670),
            ('A little cat standing on the lap of a man sitting in a chair', 563723),
        ],

    'sub-02':
        [
            ("A woman sits in a beach chair as a man walks along the sand", 406591),
            ("A little cat standing on the lap of a man sitting in a chair", 563723),
            ("A lonely giraffe is walking in the middle of a grassy field", 254568)
        ],

    'sub-03':
        [
            ("A party of young people in a bedroom with a large box of pizza", 42685),
            ("A man walking across a field of snow holding skis and ski poles", 473773),
            ("Two men are discussing something next to a desk with a laptop", 485909),
        ],

    'sub-04':
        [
            ('A young male holding a racket and tennis ball in a tennis court', 544502),
            ('A group of kids swimming in the ocean around a man on a surfboard', 556512),
            ('A cat and a dog staring intensely at each other on an armchair', 410573),
        ],

    'sub-05':
        [
            ('A man stands by a rainy street with an umbrella over his head', 48670),
            ('A woman working on her computer while also using her cell phone', 263212),
            ('An old broken down church and graffiti on surrounding structures', 214816),
        ],

    'sub-07':
        [
            ('A teddy bear lying down on the sidewalk in front of a store', 141343),
            ('A woman leaning out a window to talk to someone on the sidewal', 213506),
            ('The man on the skateboard and the dog are getting their picture taken', 162396),
        ],
}

IDS_IMAGES_IMAGERY = [scene[1] for scenes_subj in IMAGERY_SCENES.values() for scene in scenes_subj]

IDS_IMAGES_TEST = [
    3862,
    6450,
    16764,
    25902,
    38938,
    43966,
    47566,
    53580,
    55413,
    57703,
    63881,
    70426,
    79642,
    122403,
    133449,
    138529,
    146411,
    159225,
    163240,
    165419,
    165874,
    176509,
    180389,
    183210,
    186308,
    186788,
    192095,
    195406,
    201601,
    211189,
    220654,
    221313,
    238193,
    252018,
    255714,
    271844,
    275995,
    278135,
    279331,
    287434,
    292888,
    310552,
    315083,
    318108,
    323797,
    353260,
    363483,
    367120,
    380011,
    385795,
    388398,
    414373,
    423618,
    450719,
    454636,
    457249,
    466514,
    467854,
    475693,
    499733,
    505655,
    512289,
    534975,
    536798,
    546074,
    548167,
    555739,
    560282,
    567627,
    573980
]

NUM_TEST_STIMULI = len(IDS_IMAGES_TEST) * 2
INDICES_TEST_STIM_CAPTION = list(range(NUM_TEST_STIMULI // 2))
INDICES_TEST_STIM_IMAGE = list(range(NUM_TEST_STIMULI // 2, NUM_TEST_STIMULI))
IDS_TEST_STIM = np.array(IDS_IMAGES_TEST + IDS_IMAGES_TEST)

TEST_STIM_TYPES = np.array([CAPTION] * len(INDICES_TEST_STIM_CAPTION) + [IMAGE] * len(INDICES_TEST_STIM_IMAGE))

TEST_BATCH_SIZE = len(IDS_TEST_STIM)

AVG_FEATS = 'avg'
LANG_FEATS_ONLY = 'lang'
VISION_FEATS_ONLY = 'vision'
FUSED_FEATS_CLS = 'fused_cls'
FUSED_FEATS_MEAN = 'fused_mean'
MATCHED_FEATS = 'matched'
SELECT_DEFAULT = 'default'

VISION_MEAN_FEAT_KEY = "vision_features_mean"
VISION_CLS_FEAT_KEY = "vision_features_cls"

LANG_MEAN_FEAT_KEY = "lang_features_mean"
LANG_CLS_FEAT_KEY = "lang_features_cls"

FUSED_MEAN_FEAT_KEY = "fused_mean_features"
FUSED_CLS_FEAT_KEY = "fused_cls_features"

FEATURE_COMBINATION_CHOICES = [AVG_FEATS, LANG_FEATS_ONLY, VISION_FEATS_ONLY, FUSED_FEATS_CLS,
                               FUSED_FEATS_MEAN,
                               MATCHED_FEATS, SELECT_DEFAULT]

VISION_FEAT_COMBINATION_CHOICES = [VISION_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY, SELECT_DEFAULT]
LANG_FEAT_COMBINATION_CHOICES = [LANG_MEAN_FEAT_KEY, LANG_CLS_FEAT_KEY, SELECT_DEFAULT]

FEATS_NA = "n_a"

DEFAULT_FEATURES = {
    "clip": AVG_FEATS,
    "imagebind": AVG_FEATS,
    "random-imagebind": AVG_FEATS,
    "flava": AVG_FEATS,
    "blip2": AVG_FEATS,
    "visualbert": FUSED_FEATS_MEAN,
    "vilt": FUSED_FEATS_MEAN,
    "bridgetower": FUSED_FEATS_CLS,
    "bert-base-uncased": LANG_FEATS_ONLY,
    "bert-large-uncased": LANG_FEATS_ONLY,
    "llama2-7b": LANG_FEATS_ONLY,
    "llama2-13b": LANG_FEATS_ONLY,
    "mistral-7b": LANG_FEATS_ONLY,
    "mixtral-8x7b": LANG_FEATS_ONLY,
    "gpt2-small": LANG_FEATS_ONLY,
    "gpt2-medium": LANG_FEATS_ONLY,
    "gpt2-large": LANG_FEATS_ONLY,
    "gpt2-xl": LANG_FEATS_ONLY,
    "vit-b-16": VISION_FEATS_ONLY,
    "vit-l-16": VISION_FEATS_ONLY,
    "resnet-18": VISION_FEATS_ONLY,
    "resnet-50": VISION_FEATS_ONLY,
    "resnet-152": VISION_FEATS_ONLY,
    "dino-base": VISION_FEATS_ONLY,
    "dino-large": VISION_FEATS_ONLY,
    "dino-giant": VISION_FEATS_ONLY,
}

DEFAULT_VISION_FEATURES = {
    "clip": VISION_CLS_FEAT_KEY,
    "imagebind": VISION_CLS_FEAT_KEY,
    "random-imagebind": VISION_CLS_FEAT_KEY,
    "flava": VISION_MEAN_FEAT_KEY,
    "blip2": VISION_MEAN_FEAT_KEY,
    "visualbert": FEATS_NA,
    "vilt": FEATS_NA,
    "bridgetower": FEATS_NA,
    "bert-base-uncased": FEATS_NA,
    "bert-large-uncased": FEATS_NA,
    "llama2-7b": FEATS_NA,
    "llama2-13b": FEATS_NA,
    "mistral-7b": FEATS_NA,
    "mixtral-8x7b": FEATS_NA,
    "gpt2-small": FEATS_NA,
    "gpt2-medium": FEATS_NA,
    "gpt2-large": FEATS_NA,
    "gpt2-xl": FEATS_NA,
    "vit-b-16": VISION_MEAN_FEAT_KEY,
    "vit-l-16": VISION_MEAN_FEAT_KEY,
    "resnet-18": VISION_MEAN_FEAT_KEY,
    "resnet-50": VISION_MEAN_FEAT_KEY,
    "resnet-152": VISION_MEAN_FEAT_KEY,
    "dino-base": VISION_MEAN_FEAT_KEY,
    "dino-large": VISION_MEAN_FEAT_KEY,
    "dino-giant": VISION_MEAN_FEAT_KEY,
}

DEFAULT_LANG_FEATURES = {
    "clip": LANG_CLS_FEAT_KEY,
    "imagebind": LANG_CLS_FEAT_KEY,
    "random-imagebind": LANG_CLS_FEAT_KEY,
    "flava": LANG_MEAN_FEAT_KEY,
    "blip2": LANG_MEAN_FEAT_KEY,
    "visualbert": FEATS_NA,
    "vilt": FEATS_NA,
    "bridgetower": FEATS_NA,
    "bert-base-uncased": LANG_MEAN_FEAT_KEY,
    "bert-large-uncased": LANG_MEAN_FEAT_KEY,
    "llama2-7b": LANG_MEAN_FEAT_KEY,
    "llama2-13b": LANG_MEAN_FEAT_KEY,
    "mistral-7b": LANG_MEAN_FEAT_KEY,
    "mixtral-8x7b": LANG_MEAN_FEAT_KEY,
    "gpt2-small": LANG_MEAN_FEAT_KEY,
    "gpt2-medium": LANG_MEAN_FEAT_KEY,
    "gpt2-large": LANG_MEAN_FEAT_KEY,
    "gpt2-xl": LANG_MEAN_FEAT_KEY,
    "vit-b-16": FEATS_NA,
    "vit-l-16": FEATS_NA,
    "resnet-18": FEATS_NA,
    "resnet-50": FEATS_NA,
    "resnet-152": FEATS_NA,
    "dino-base": FEATS_NA,
    "dino-large": FEATS_NA,
    "dino-giant": FEATS_NA,
}


@dataclass
class LatentFeatsConfig:
    model: str
    features: str
    test_features: str
    vision_features: str
    lang_features: str
    logging: bool = True

    def __post_init__(self):
        if self.features == SELECT_DEFAULT:
            self.features = DEFAULT_FEATURES[self.model]
            if logging:
                print(f"Selected default features for {self.model}: {self.features}")
        if self.test_features == SELECT_DEFAULT:
            self.test_features = DEFAULT_FEATURES[self.model]
            if logging:
                print(f"Selected default test features for {self.model}: {self.test_features}")
        if self.vision_features == SELECT_DEFAULT:
            self.vision_features = DEFAULT_VISION_FEATURES[self.model]
            if logging:
                print(f"Selected default vision features for {self.model}: {self.vision_features}")
        if self.lang_features == SELECT_DEFAULT:
            self.lang_features = DEFAULT_LANG_FEATURES[self.model]
            if logging:
                print(f"Selected default language features for {self.model}: {self.lang_features}")

        self.combined_feats = f"{self.features}_test_{self.test_features}"


def features_config_from_combined_features(model, combined_feats, vision_features, lang_features, logging=True):
    train_feats, test_feats = combined_feats.split("_")
    return LatentFeatsConfig(model, train_feats, test_feats, vision_features, lang_features, logging)


def stim_id_from_beta_file_name(beta_file_name):
    return int(beta_file_name.replace('beta_I', '').replace('beta_C', '').replace('beta_', '').replace(".nii", ''))


def get_fmri_data_paths(betas_dir, subject, mode, surface=False, hemi=None, resolution=None):
    if surface:
        if hemi is None or resolution is None:
            raise ValueError("Hemi and resolution needs to be specified to load surface-level data")
        filename_suffix = f"*_{hemi}.gii"
        fmri_addresses_regex = os.path.join(
            FMRI_SURFACE_LEVEL_DIR, resolution, subject, f'betas_{mode}*', filename_suffix
        )
    else:
        filename_suffix = "*.nii"
        fmri_addresses_regex = os.path.join(betas_dir, subject, f'betas_{mode}*', filename_suffix)

    fmri_betas_paths = sorted(glob(fmri_addresses_regex))
    stim_ids = []
    stim_types = []
    for path in fmri_betas_paths:
        split_name = path.split(os.sep)[-2]
        stim_id = stim_id_from_beta_file_name(os.path.basename(path))
        if IMAGERY in split_name:
            stim_types.append(IMAGERY)
            stim_id = IMAGERY_SCENES[subject][stim_id - 1][1]
        elif IMAGE in split_name:
            stim_types.append(IMAGE)
        elif CAPTION in split_name:
            stim_types.append(CAPTION)
        else:
            raise RuntimeError(f"Unknown split name: {split_name}")

        stim_ids.append(stim_id)

    stim_ids = np.array(stim_ids)
    stim_types = np.array(stim_types)

    return fmri_betas_paths, stim_ids, stim_types


def get_fmri_betas_scaler_path(subject, mode, mask_name=None):
    scaler_file_name = f'betas_scaler_{mode}.p'
    if mask_name is not None:
        scaler_file_name += f'_mask_{mask_name}'
    return os.path.join(FMRI_NORMALIZATIONS_DIR, "normalizations", subject, scaler_file_name)


def get_fmri_betas_nan_locations_path(subject):
    return os.path.join(FMRI_NORMALIZATIONS_DIR, "normalizations", subject, 'betas_nan_locations.p')


def get_fmri_betas_nan_locations(betas_dir, subject, training_mode, split, latent_feats_config):
    nan_locations_path = get_fmri_betas_nan_locations_path(subject)
    if not os.path.isfile(nan_locations_path):
        os.makedirs(os.path.dirname(nan_locations_path), exist_ok=True)
        graymatter_mask = load_graymatter_mask(subject)
        latent_features = pickle.load(open(model_features_file_path(latent_feats_config.model_name), 'rb'))
        ds = DecodingDataset(betas_dir, subject, training_mode, split, latent_features, latent_feats_config,
                             graymatter_mask)
        if split == SPLIT_TRAIN:
            fmri_beta = next(iter(ds))
        else:
            fmri_beta, _, _, _ = next(iter(ds))

        nan_locations = np.isnan(fmri_beta)
        pickle.dump(nan_locations, open(nan_locations_path, 'wb'))

    nan_locations = pickle.load(open(nan_locations_path, 'rb'))
    return nan_locations


def get_fmri_betas_standardization_scaler(betas_dir, subject, training_mode, split, latent_feats_config):
    scaler_path = get_fmri_betas_scaler_path(subject, training_mode)
    if not os.path.isfile(scaler_path):
        print(f"Calculating mean and std over whole {training_mode} set betas for standardization.")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        graymatter_mask = load_graymatter_mask(subject)
        latent_features = pickle.load(open(model_features_file_path(latent_feats_config.model_name), 'rb'))
        ds = DecodingDataset(betas_dir, subject, training_mode, split, latent_features, latent_feats_config,
                             graymatter_mask)
        if split == SPLIT_TRAIN:
            fmri_betas = [beta for beta, _ in tqdm(iter(ds), total=len(ds))]
        else:
            fmri_betas = [beta for beta, _, _, _ in tqdm(iter(ds), total=len(ds))]

        scaler = StandardScaler().fit(fmri_betas)

        pickle.dump(scaler, open(scaler_path, 'wb'))

    scaler = pickle.load(open(scaler_path, 'rb'))
    return scaler


def get_latent_feats(all_latents, stim_id, stim_type, latent_feats_config):
    if latent_feats_config.features == VISION_FEATS_ONLY:
        feats = get_vision_feats(all_latents, stim_id, latent_feats_config.vision_features)
    elif latent_feats_config.features == LANG_FEATS_ONLY:
        feats = get_lang_feats(all_latents, stim_id, latent_feats_config.lang_features)
    elif latent_feats_config.features == AVG_FEATS:
        vision_feats = get_vision_feats(all_latents, stim_id, latent_feats_config.vision_features)
        lang_feats = get_lang_feats(all_latents, stim_id, latent_feats_config.lang_features)
        feats = np.stack((lang_feats, vision_feats))
        feats = feats.mean(axis=0)
    elif latent_feats_config.features == FUSED_FEATS_CLS:
        feats = all_latents[stim_id][FUSED_CLS_FEAT_KEY]
    elif latent_feats_config.features == FUSED_FEATS_MEAN:
        feats = all_latents[stim_id][FUSED_MEAN_FEAT_KEY]
    elif latent_feats_config.features == MATCHED_FEATS:
        if stim_type == CAPTION:
            feats = get_lang_feats(all_latents, stim_id, latent_feats_config.lang_features)
        elif stim_type == IMAGE:
            feats = get_vision_feats(all_latents, stim_id, latent_feats_config.vision_features)
        elif stim_type == IMAGERY:
            feats = get_vision_feats(all_latents, stim_id, latent_feats_config.vision_features)
        else:
            raise RuntimeError(f"Unknown stim type: {stim_type}")
    else:
        raise RuntimeError(f"Unknown feature selection/combination method: {latent_feats_config.features}")

    return feats


def get_fmri_surface_data(subject, mode, resolution, hemis=HEMIS):
    base_mode = mode.split('_')[0]
    fmri_betas = {
        hemi: pickle.load(
            open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_{hemi}_{resolution}_{base_mode}.p"), 'rb')) for hemi
        in tqdm(hemis, desc=f"loading {base_mode} fmri surface data")
    }
    stim_ids = pickle.load(open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_stim_ids_{base_mode}.p"), 'rb'))
    stim_types = pickle.load(open(os.path.join(FMRI_SURFACE_LEVEL_DIR, f"{subject}_stim_types_{base_mode}.p"), 'rb'))

    if mode == MOD_SPECIFIC_CAPTIONS:
        for hemi in hemis:
            fmri_betas[hemi] = fmri_betas[hemi][stim_types == CAPTION]
        stim_ids = stim_ids[stim_types == CAPTION]
        stim_types = stim_types[stim_types == CAPTION]
    elif mode == MOD_SPECIFIC_IMAGES:
        for hemi in hemis:
            fmri_betas[hemi] = fmri_betas[hemi][stim_types == IMAGE]
        stim_ids = stim_ids[stim_types == IMAGE]
        stim_types = stim_types[stim_types == IMAGE]

    return fmri_betas, stim_ids, stim_types


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


def get_latent_features(model_name, feats_config, stim_ids, stim_types, test_mode=False):
    latent_vectors_file = model_features_file_path(model_name)
    latent_vectors = pickle.load(open(latent_vectors_file, 'rb'))

    features = feats_config.test_features if test_mode else feats_config.features
    nn_latent_vectors = []
    for stim_id, stim_type in zip(stim_ids, stim_types):
        if features == VISION_FEATS_ONLY:
            feats = get_vision_feats(latent_vectors, stim_id, feats_config.vision_features)
        elif features == LANG_FEATS_ONLY:
            feats = get_lang_feats(latent_vectors, stim_id, feats_config.lang_features)
        elif features == AVG_FEATS:
            vision_feats = get_vision_feats(latent_vectors, stim_id, feats_config.vision_features)
            lang_feats = get_lang_feats(latent_vectors, stim_id, feats_config.lang_features)
            feats = np.stack((lang_feats, vision_feats))
            feats = feats.mean(axis=0)
        elif features == FUSED_FEATS_CLS:
            feats = latent_vectors[stim_id][FUSED_CLS_FEAT_KEY]
        elif features == FUSED_FEATS_MEAN:
            feats = latent_vectors[stim_id][FUSED_MEAN_FEAT_KEY]
        elif features == MATCHED_FEATS:
            if stim_type == CAPTION:
                feats = get_lang_feats(latent_vectors, stim_id, feats_config.lang_features)
            elif stim_type == IMAGE:
                feats = get_vision_feats(latent_vectors, stim_id, feats_config.vision_features)
            elif stim_type == IMAGERY:
                feats = get_vision_feats(latent_vectors, stim_id, feats_config.vision_features)
            else:
                raise RuntimeError(f"Unknown stim type: {stim_type}")
        else:
            raise RuntimeError(f"Unknown feature selection/combination method: {features}")
        nn_latent_vectors.append(feats)

    nn_latent_vectors = np.array(nn_latent_vectors, dtype=np.float32)

    return nn_latent_vectors


def get_fmri_voxel_data(betas_dir, subject, mode):
    fmri_betas_paths, stim_ids, stim_types = get_fmri_data_paths(betas_dir, subject, mode)

    graymatter_mask = load_graymatter_mask(subject)
    fmri_betas = []
    for idx in trange(len(fmri_betas_paths), desc="loading fmri data"):
        sample = nib.load(fmri_betas_paths[idx])
        sample = sample.get_fdata()
        sample = sample[graymatter_mask].astype('float32').reshape(-1)
        fmri_betas.append(sample)

    fmri_betas = np.array(fmri_betas)
    return fmri_betas, stim_ids, stim_types


def get_fmri_data(betas_dir, subject, mode, surface=False, resolution=None):
    if surface:
        fmri_betas, stim_ids, stim_types = get_fmri_surface_data(subject, mode, resolution)
    else:
        fmri_betas, stim_ids, stim_types = get_fmri_voxel_data(betas_dir, subject, mode)

    return fmri_betas, stim_ids, stim_types


def get_latents_scaler_path(subject, feats_config, mode):
    scaler_file_name = (
        f'scaler_{feats_config.model_name}_{feats_config.features}_{feats_config.vision_features}_'
        f'{feats_config.lang_features}_{mode}.p')
    return os.path.join(LATENT_FEATURES_NORMALIZATIONS_DIR, subject, scaler_file_name)


def standardize_latents(train_latents, test_latents, imagery_latents=None):
    scaler = StandardScaler()
    scaler.fit(train_latents)
    train_latents = scaler.transform(train_latents)
    test_latents = scaler.transform(test_latents)

    if imagery_latents is not None:
        imagery_latents = scaler.transform(imagery_latents)
        return train_latents, test_latents, imagery_latents

    return train_latents, test_latents


def standardize_fmri_betas(train_fmri_betas, test_fmri_betas, imagery_fmri_betas=None):
    # print("before standardization:")
    # print(f'train fmri mean: {train_fmri_betas.mean(axis=0).mean():.2f}')
    # print(f'test fmri mean: {test_fmri_betas.mean(axis=0).mean():.2f}')
    # print(f'train fmri stddev: {train_fmri_betas.std(axis=0).mean():.2f}')
    # print(f'test fmri stddev: {test_fmri_betas.std(axis=0).mean():.2f}')

    scaler = StandardScaler()
    scaler.fit(train_fmri_betas)

    train_fmri_betas = scaler.transform(train_fmri_betas)

    # test_scaler = StandardScaler()
    # test_scaler.fit(test_fmri_betas)
    test_fmri_betas = scaler.transform(test_fmri_betas)

    # print("after standardization:")
    # print(f'train fmri mean: {train_fmri_betas.mean(axis=0).mean():.2f}')
    # print(f'test fmri mean: {test_fmri_betas.mean(axis=0).mean():.2f}')
    # print(f'train fmri stddev: {train_fmri_betas.std(axis=0).mean():.2f}')
    # print(f'test fmri stddev: {test_fmri_betas.std(axis=0).mean():.2f}')

    if imagery_fmri_betas is not None:
        imagery_fmri_betas = scaler.transform(imagery_fmri_betas)
        return train_fmri_betas, test_fmri_betas, imagery_fmri_betas

    return train_fmri_betas, test_fmri_betas


def get_latent_feats_standardization_scaler(betas_dir, subject, latent_feats_config, training_mode):
    scaler_path = get_latents_scaler_path(subject, latent_feats_config, training_mode)
    if not os.path.isfile(scaler_path):
        print("Calculating mean and std over whole train set latents for standardization.")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        graymatter_mask = load_graymatter_mask(subject)
        latent_features = pickle.load(open(model_features_file_path(latent_feats_config.model_name), 'rb'))
        train_ds = DecodingDataset(betas_dir, subject, training_mode, SPLIT_TRAIN, latent_features, latent_feats_config,
                                   graymatter_mask)
        train_latents = np.array([latents for _, latents in tqdm(iter(train_ds), total=len(train_ds))])

        scaler = StandardScaler().fit(train_latents)

        pickle.dump(scaler, open(scaler_path, 'wb'), pickle.HIGHEST_PROTOCOL)

    scaler = pickle.load(open(scaler_path, 'rb'))

    return scaler


class DecodingDataset(Dataset):
    def __init__(self, betas_dir, subject, mode, split, latent_features, latent_feats_config, graymatter_mask,
                 betas_transform=None, latent_feats_transform=None, betas_nan_locations=None):
        self.data_paths, self.stim_ids, self.stim_types = get_fmri_data_paths(betas_dir, subject, mode, split)
        self.betas_transform = betas_transform
        self.betas_nan_locations = betas_nan_locations
        self.graymatter_mask = graymatter_mask

        self.latent_features = latent_features
        self.latent_feats_config = latent_feats_config
        self.latent_feats_transform = latent_feats_transform

        self.split = split

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        betas = nib.load(self.data_paths[idx])
        betas = betas.get_fdata()

        if self.graymatter_mask is not None:
            betas = betas[self.graymatter_mask]

        if self.betas_nan_locations is not None:
            betas = betas[~self.betas_nan_locations]

        betas = betas.astype('float32').reshape(-1)
        if self.betas_transform is not None:
            betas = self.betas_transform(betas)

        betas = np.nan_to_num(betas)

        stim_id = self.stim_ids[idx]
        stim_type = self.stim_types[idx]

        latents = get_latent_feats(self.latent_features, stim_id, stim_type, self.latent_feats_config)
        if self.latent_feats_transform is not None:
            latents = self.latent_feats_transform(latents)

        if self.split == SPLIT_TRAIN:
            return betas, latents
        else:
            return betas, latents, stim_type, stim_id


def remove_nans(betas_list):
    nan_locations = np.logical_or.reduce([np.isnan(betas[0]) for betas in betas_list])
    betas_list = [betas[:, ~nan_locations] for betas in betas_list]
    return betas_list


def create_null_distr_seeds(n_permutations_per_subject):
    random_seeds = []
    seed = 0
    for _ in range(n_permutations_per_subject):
        # shuffle indices for captions and images separately until all indices have changed
        shuffled_indices = create_shuffled_indices(seed)
        while any(shuffled_indices == np.arange(NUM_TEST_STIMULI)):
            seed += 1
            shuffled_indices = create_shuffled_indices(seed)
        random_seeds.append(seed)
        seed += 1
    return random_seeds


def create_shuffled_indices(seed):
    np.random.seed(seed)
    num_stim_one_mod = NUM_TEST_STIMULI // 2
    shuffleidx_mod_1 = np.random.choice(range(num_stim_one_mod), size=num_stim_one_mod,
                                        replace=False)
    shuffleidx_mod_2 = np.random.choice(range(num_stim_one_mod, NUM_TEST_STIMULI),
                                        size=num_stim_one_mod, replace=False)
    return np.concatenate((shuffleidx_mod_1, shuffleidx_mod_2))


def apply_mask(mask_name, betas_list, args):
    if mask_name is not None:
        if not args.surface:
            raise NotImplementedError()
        mask = pickle.load(open(mask_name, 'rb'))
        mask_flat = np.concatenate((mask[HEMIS[0]], mask[HEMIS[1]]))
        betas_list = [betas[:, mask_flat == 1].copy() for betas in betas_list]

    return betas_list


def load_graymatter_mask(subject):
    gray_matter_mask_path = get_graymatter_mask_path(subject)
    gray_matter_mask_img = nib.load(gray_matter_mask_path)
    gray_matter_mask_data = gray_matter_mask_img.get_fdata()
    gray_matter_mask = gray_matter_mask_data == 1
    print(f"Gray matter mask size: {gray_matter_mask.sum()}")
    return gray_matter_mask


class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, betas_dir, batch_size, subject, training_mode, feats_config, num_workers, cv_split=0,
                 num_cv_splits=5):
        super().__init__()
        assert cv_split < num_cv_splits

        self.betas_dir = betas_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.subject = subject
        self.training_mode = training_mode
        self.feats_config = feats_config

        self.graymatter_mask = load_graymatter_mask(subject)

        self.latents_transform = get_latent_feats_standardization_scaler(betas_dir, subject, feats_config,
                                                                         training_mode)

        self.betas_transform = get_fmri_betas_standardization_scaler(betas_dir, subject, training_mode, SPLIT_TRAIN,
                                                                     feats_config)

        self.betas_nan_locations = get_fmri_betas_nan_locations(betas_dir, subject, training_mode, SPLIT_TRAIN,
                                                                feats_config)
        print("Loading pickle with latent feats..", end=" ")
        latent_features = pickle.load(open(model_features_file_path(feats_config.model_name), 'rb'))
        print("done.")

        self.data = DecodingDataset(
            self.betas_dir,
            self.subject, self.training_mode, SPLIT_TRAIN, latent_features, feats_config, self.graymatter_mask,
            self.betas_transform, self.latents_transform, self.betas_nan_locations,
        )
        indices = list(range(len(self.data)))
        val_split_size = round(len(self.data) / num_cv_splits)
        val_indices = indices[cv_split * val_split_size: (cv_split + 1) * val_split_size]
        train_indices = [i for i in indices if i not in val_indices]
        self.ds_train = Subset(self.data, train_indices)
        self.ds_val = Subset(self.data, val_indices)
        self.ds_test = DecodingDataset(
            self.betas_dir,
            self.subject, TESTING_MODE, SPLIT_TEST, latent_features, feats_config, self.graymatter_mask,
            self.betas_transform, self.latents_transform, self.betas_nan_locations,
        )
        self.ds_imagery = DecodingDataset(
            self.betas_dir,
            self.subject, IMAGERY, SPLIT_IMAGERY, latent_features, feats_config, self.graymatter_mask,
            self.betas_transform, self.latents_transform, self.betas_nan_locations,
        )

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=TEST_BATCH_SIZE, num_workers=self.num_workers)
