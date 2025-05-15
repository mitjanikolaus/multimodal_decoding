import logging
import os
import pickle
from dataclasses import dataclass
from glob import glob

import nibabel
import numpy as np
from sklearn.preprocessing import StandardScaler
import nibabel as nib
from tqdm import trange

from utils import model_features_file_path, HEMIS, DEFAULT_RESOLUTION, FMRI_STIM_INFO_DIR, FMRI_BETAS_DIR, \
    FS_HEMI_NAMES, FREESURFER_HOME_DIR

MODALITY_SPECIFIC_IMAGES = "images"
MODALITY_SPECIFIC_CAPTIONS = "captions"
MODALITY_AGNOSTIC = "agnostic"
TRAINING_MODES = [MODALITY_AGNOSTIC, MODALITY_SPECIFIC_CAPTIONS, MODALITY_SPECIFIC_IMAGES]

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
SPLIT_IMAGERY = "imagery"

SPLIT_TEST_IMAGES = "test_image"
SPLIT_TEST_CAPTIONS = "test_caption"

SPLIT_TEST_IMAGE_ATTENDED = "test_image_attended"
SPLIT_TEST_CAPTION_ATTENDED = "test_caption_attended"
SPLIT_TEST_IMAGE_UNATTENDED = "test_image_unattended"
SPLIT_TEST_CAPTION_UNATTENDED = "test_caption_unattended"
SPLIT_IMAGERY_WEAK = "imagery_weak"

TEST_SPLITS = [SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS, SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED,
               SPLIT_TEST_IMAGE_UNATTENDED, SPLIT_TEST_CAPTION_UNATTENDED, SPLIT_IMAGERY, SPLIT_IMAGERY_WEAK]
ALL_SPLITS = [SPLIT_TRAIN] + TEST_SPLITS
ALL_SPLITS_BASE_DATA = [SPLIT_TRAIN, SPLIT_TEST_IMAGES, SPLIT_TEST_CAPTIONS, SPLIT_IMAGERY]

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
     'A woman leaning out a window to talk to someone on the sidewalk'],
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
            ('A woman leaning out a window to talk to someone on the sidewalk', 213506),
            ('The man on the skateboard and the dog are getting their picture taken', 162396),
        ],
}

IMAGERY_STIMS_IDS = {sub: [scenes_subj[1] for scenes_subj in scene] for sub, scene in IMAGERY_SCENES.items()}
IMAGERY_STIMS_TYPES = {sub: [IMAGERY for scenes_subj in scene] for sub, scene in IMAGERY_SCENES.items()}

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

TEST_STIM_IDS = np.array(IDS_IMAGES_TEST + IDS_IMAGES_TEST)
TEST_STIM_TYPES = np.array([CAPTION] * len(INDICES_TEST_STIM_CAPTION) + [IMAGE] * len(INDICES_TEST_STIM_IMAGE))

TEST_BATCH_SIZE = len(TEST_STIM_IDS)

IDS_IMAGES_IMAGERY_WEAK = [
    9800,
    11288,
    15645,
    22563,
    26538,
    29776,
    51079,
    51525,
    63214,
    68597,
    88045,
    112022,
    120858,
    126305,
    127078,
    129362,
    141468,
    143054,
    179199,
    198508,
    215939,
    229947,
    238598,
    263047,
    271195,
    271417,
    273443,
    273840,
    283254,
    284623,
    286708,
    287049,
    292622,
    311377,
    323599,
    325277,
    332866,
    338384,
    382320,
    416749,
    420400,
    421457,
    429761,
    464382,
    495850,
    516344,
    517685,
    561013,
    563194,
    567838,
]

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

FEATURE_COMBINATION_CHOICES = [
    AVG_FEATS, LANG_FEATS_ONLY, VISION_FEATS_ONLY, FUSED_FEATS_CLS, FUSED_FEATS_MEAN, MATCHED_FEATS, SELECT_DEFAULT
]

VISION_FEAT_COMBINATION_CHOICES = [VISION_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY, SELECT_DEFAULT]
LANG_FEAT_COMBINATION_CHOICES = [LANG_MEAN_FEAT_KEY, LANG_CLS_FEAT_KEY, SELECT_DEFAULT]

FEATS_NA = "n_a"

DEFAULT_FEATURES = {
    "siglip": AVG_FEATS,
    "paligemma2": AVG_FEATS,
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
    "vit-h-14": VISION_FEATS_ONLY,
    "resnet-18": VISION_FEATS_ONLY,
    "resnet-50": VISION_FEATS_ONLY,
    "resnet-152": VISION_FEATS_ONLY,
    "dino-base": VISION_FEATS_ONLY,
    "dino-large": VISION_FEATS_ONLY,
    "dino-giant": VISION_FEATS_ONLY,
    "gabor": MATCHED_FEATS,
}

DEFAULT_VISION_FEATURES = {
    "siglip": VISION_CLS_FEAT_KEY,
    "paligemma2": VISION_MEAN_FEAT_KEY,
    "clip": VISION_CLS_FEAT_KEY,
    "imagebind": VISION_CLS_FEAT_KEY,
    "random-imagebind": VISION_CLS_FEAT_KEY,
    "flava": VISION_CLS_FEAT_KEY,
    "blip2": VISION_CLS_FEAT_KEY,
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
    "vit-h-14": VISION_MEAN_FEAT_KEY,
    "resnet-18": VISION_MEAN_FEAT_KEY,
    "resnet-50": VISION_MEAN_FEAT_KEY,
    "resnet-152": VISION_MEAN_FEAT_KEY,
    "dino-base": VISION_MEAN_FEAT_KEY,
    "dino-large": VISION_MEAN_FEAT_KEY,
    "dino-giant": VISION_MEAN_FEAT_KEY,
    "gabor": VISION_CLS_FEAT_KEY,
}

DEFAULT_LANG_FEATURES = {
    "siglip": LANG_CLS_FEAT_KEY,
    "paligemma2": LANG_MEAN_FEAT_KEY,
    "clip": LANG_CLS_FEAT_KEY,
    "imagebind": LANG_CLS_FEAT_KEY,
    "random-imagebind": LANG_CLS_FEAT_KEY,
    "flava": LANG_CLS_FEAT_KEY,
    "blip2": LANG_CLS_FEAT_KEY,
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
    "vit-h-14": FEATS_NA,
    "resnet-18": FEATS_NA,
    "resnet-50": FEATS_NA,
    "resnet-152": FEATS_NA,
    "dino-base": FEATS_NA,
    "dino-large": FEATS_NA,
    "dino-giant": FEATS_NA,
    "gabor": LANG_CLS_FEAT_KEY,
}


@dataclass
class LatentFeatsConfig:
    model: str
    features: str = SELECT_DEFAULT
    test_features: str = SELECT_DEFAULT
    vision_features: str = SELECT_DEFAULT
    lang_features: str = SELECT_DEFAULT
    logging: bool = True

    def __post_init__(self):
        if self.features == SELECT_DEFAULT:
            self.features = DEFAULT_FEATURES[self.model]
        if self.test_features == SELECT_DEFAULT:
            self.test_features = DEFAULT_FEATURES[self.model]
        if self.vision_features == SELECT_DEFAULT:
            self.vision_features = DEFAULT_VISION_FEATURES[self.model]
        if self.lang_features == SELECT_DEFAULT:
            self.lang_features = DEFAULT_LANG_FEATURES[self.model]
        if logging:
            print(f"Selected features for {self.model}: {self.features} {self.test_features} "
                  f"{self.vision_features} {self.lang_features}")
        self.combined_feats = f"{self.features}_test_{self.test_features}"


def stim_id_from_beta_file_name(beta_file_name, suffix='.nii'):
    return int(beta_file_name.replace('beta_I', '').replace('beta_C', '').replace('beta_', '').replace(suffix, ''))


def get_fmri_data_paths(betas_dir, subject, split, mode=MODALITY_AGNOSTIC, hemi=None, suffix='.nii'):
    mode_suffix = ""
    if mode == MODALITY_SPECIFIC_CAPTIONS:
        mode_suffix = f"_{CAPTION}"
    elif mode == MODALITY_SPECIFIC_IMAGES:
        mode_suffix = f"_{IMAGE}"

    if hemi is None:
        fmri_addresses_regex = os.path.join(betas_dir, subject, f'betas_{split}{mode_suffix}*', f"*{suffix}")
    else:
        fmri_addresses_regex = os.path.join(betas_dir, hemi, subject, f'betas_{split}{mode_suffix}*', f"*{suffix}")
    fmri_betas_paths = sorted(glob(fmri_addresses_regex))

    stim_ids = []
    stim_types = []
    for path in fmri_betas_paths:
        split_name = path.split(os.sep)[-2]
        stim_id = stim_id_from_beta_file_name(os.path.basename(path), suffix)
        if SPLIT_IMAGERY_WEAK in split_name:
            stim_types.append(IMAGERY)
        elif IMAGERY in split_name:
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


def get_stim_info(subject, split):
    if split == SPLIT_TRAIN:
        stim_ids = pickle.load(open(os.path.join(FMRI_STIM_INFO_DIR, f"{subject}_stim_ids_{split}.p"), 'rb'))
        stim_types = pickle.load(open(os.path.join(FMRI_STIM_INFO_DIR, f"{subject}_stim_types_{split}.p"), 'rb'))
    elif split == SPLIT_TEST_IMAGES:
        stim_ids, stim_types = IDS_IMAGES_TEST, [IMAGE for _ in TEST_STIM_IDS]
    elif split == SPLIT_TEST_CAPTIONS:
        stim_ids, stim_types = IDS_IMAGES_TEST, [CAPTION for _ in TEST_STIM_IDS]
    elif split == SPLIT_IMAGERY:
        stim_ids, stim_types = IMAGERY_STIMS_IDS[subject], IMAGERY_STIMS_TYPES[subject]
    elif split == SPLIT_IMAGERY_WEAK:
        stim_ids, stim_types = IDS_IMAGES_IMAGERY_WEAK, [IMAGERY for _ in IDS_IMAGES_IMAGERY_WEAK]
    elif split in [SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_IMAGE_UNATTENDED]:
        stim_ids, stim_types = IDS_IMAGES_TEST, [IMAGE for _ in IDS_IMAGES_TEST]
    elif split in [SPLIT_TEST_CAPTION_ATTENDED, SPLIT_TEST_CAPTION_UNATTENDED]:
        stim_ids, stim_types = IDS_IMAGES_TEST, [CAPTION for _ in IDS_IMAGES_TEST]
    else:
        raise RuntimeError(f"Unknown split name: {split}")

    return stim_ids, stim_types


def get_latent_features(feats_config, subject, split, mode=MODALITY_AGNOSTIC):
    latent_vectors_file = model_features_file_path(feats_config.model)
    latent_vectors = pickle.load(open(latent_vectors_file, 'rb'))

    stim_ids, stim_types = get_stim_info(subject, split)

    if mode == MODALITY_SPECIFIC_CAPTIONS:
        stim_ids = stim_ids[stim_types == CAPTION]
        stim_types = stim_types[stim_types == CAPTION]
    elif mode == MODALITY_SPECIFIC_IMAGES:
        stim_ids = stim_ids[stim_types == IMAGE]
        stim_types = stim_types[stim_types == IMAGE]

    features = feats_config.test_features if split in [SPLIT_TEST, SPLIT_IMAGERY] else feats_config.features
    nn_latent_vectors = []
    for i, stim_id in enumerate(stim_ids):
        if features == VISION_FEATS_ONLY:
            feats = get_vision_feats(latent_vectors, stim_id, feats_config.vision_features)
        elif features == LANG_FEATS_ONLY:
            feats = get_lang_feats(latent_vectors, stim_id, feats_config.lang_features)
        elif features == AVG_FEATS:
            vision_feats = get_vision_feats(latent_vectors, stim_id, feats_config.vision_features)
            lang_feats = get_lang_feats(latent_vectors, stim_id, feats_config.lang_features)
            feats = np.mean((lang_feats, vision_feats), axis=0)
        elif features == FUSED_FEATS_CLS:
            feats = latent_vectors[stim_id][FUSED_CLS_FEAT_KEY]
        elif features == FUSED_FEATS_MEAN:
            feats = latent_vectors[stim_id][FUSED_MEAN_FEAT_KEY]
        elif features == MATCHED_FEATS:
            stim_type = stim_types[i]
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


def get_fmri_surface_data(betas_dir, subject, split, mode=MODALITY_AGNOSTIC, hemi=HEMIS[0]):
    fmri_betas_paths, stim_ids, stim_types = get_fmri_data_paths(betas_dir, subject, split, mode, hemi, suffix='.gii')

    fmri_betas = []
    for idx in trange(len(fmri_betas_paths), desc=f"loading {subject} {mode} {hemi} hemi {split} fmri data"):
        sample = nib.load(fmri_betas_paths[idx])
        sample = sample.darrays[0].data
        fmri_betas.append(sample)

    fmri_betas = np.array(fmri_betas)
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


def get_fmri_data(betas_dir, subject, split, mode=MODALITY_AGNOSTIC, surface=False, resolution=DEFAULT_RESOLUTION):
    if surface:
        betas_dir = os.path.join(betas_dir, "surface")
        betas_left_hemi, stim_ids, stim_types = get_fmri_surface_data(betas_dir, subject, split, mode, "left")
        betas_right_hemi, _, _ = get_fmri_surface_data(betas_dir, subject, split, mode, "right")

        betas = np.hstack((betas_left_hemi, betas_right_hemi))
        return betas, stim_ids, stim_types
    else:
        return get_fmri_voxel_data(betas_dir, subject, split, mode)


def get_fmri_voxel_data(betas_dir, subject, split, mode=MODALITY_AGNOSTIC):
    fmri_betas_paths, stim_ids, stim_types = get_fmri_data_paths(betas_dir, subject, split, mode)

    fmri_betas = []
    for idx in trange(len(fmri_betas_paths), desc=f"loading {subject} {split} fmri data"):
        sample = nib.load(fmri_betas_paths[idx])
        sample = sample.get_fdata().astype('float32').reshape(-1)
        fmri_betas.append(sample)

    fmri_betas = np.array(fmri_betas)
    return fmri_betas, stim_ids, stim_types


def standardize_latents(latents):
    scaler = StandardScaler().fit(latents[SPLIT_TRAIN])
    latents = {split: scaler.transform(lat).astype(np.float32) for split, lat in latents.items()}

    return latents


def standardize_fmri_betas(fmri_betas):
    nan_locations = np.isnan(fmri_betas[SPLIT_TRAIN][0])
    print(f"Ignoring data from {np.sum(nan_locations)} nan locations.")
    fmri_betas = {split: betas[:, ~nan_locations] for split, betas in fmri_betas.items()}

    scaler = StandardScaler().fit(fmri_betas[SPLIT_TRAIN])

    fmri_betas = {split: scaler.transform(betas).astype(np.float32) for split, betas in fmri_betas.items()}

    return fmri_betas


def create_null_distr_shuffled_indices(n_permutations_per_subject):
    shuffled_indices = []
    seed = 0
    for _ in range(n_permutations_per_subject):
        # shuffle indices for captions and images separately until all indices have changed
        indices = create_shuffled_indices(seed)
        while any(indices == np.arange(NUM_TEST_STIMULI)):
            seed += 1
            indices = create_shuffled_indices(seed)
        shuffled_indices.append(indices)
        seed += 1
    return shuffled_indices


def create_shuffled_indices(seed):
    np.random.seed(seed)
    num_stim_one_mod = NUM_TEST_STIMULI // 2
    shuffleidx_mod_1 = np.random.choice(range(num_stim_one_mod), size=num_stim_one_mod,
                                        replace=False)
    shuffleidx_mod_2 = np.random.choice(range(num_stim_one_mod, NUM_TEST_STIMULI),
                                        size=num_stim_one_mod, replace=False)
    return np.concatenate((shuffleidx_mod_1, shuffleidx_mod_2))


def apply_mask(mask, fmri_betas, args):
    if mask is not None:
        if not args.surface:
            raise NotImplementedError("The --surface option needs to be specified when using masks")
        if os.path.isfile(mask):
            mask = pickle.load(open(mask, 'rb'))
            mask_flat = np.concatenate((mask[HEMIS[0]], mask[HEMIS[1]]))
        else:
            masks_hemis = dict()
            for hemi in HEMIS:
                roi_names = mask.split('_')
                roi_names_hemi = [name.split('-')[1] for name in roi_names if name.split('-')[0] == hemi]
                hemi_fs = FS_HEMI_NAMES[hemi]
                atlas_path = os.path.join(FREESURFER_HOME_DIR, f"subjects/fsaverage/label/{hemi_fs}.aparc.annot")
                atlas_labels, _, names = nibabel.freesurfer.read_annot(atlas_path)
                names = [name.decode() for name in names]
                regions_indices = [names.index(roi) for roi in roi_names_hemi]
                masks_hemis[hemi] = np.array([1 if l in regions_indices else 0 for l in atlas_labels])
            mask_flat = np.concatenate((masks_hemis[HEMIS[0]], masks_hemis[HEMIS[1]]))

        fmri_betas = {split: betas[:, mask_flat == 1].copy() for split, betas in fmri_betas.items()}

    return fmri_betas
