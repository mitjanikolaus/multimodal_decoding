import os
import pickle
from dataclasses import dataclass
from glob import glob

import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import nibabel as nib
from tqdm import tqdm

from analyses.ridge_regression_decoding import IMAGERY, IMAGE, CAPTION, \
    get_vision_feats, VISION_FEATS_ONLY, LANG_FEATS_ONLY, get_lang_feats, AVG_FEATS, \
    FUSED_FEATS_CLS, FUSED_FEATS_MEAN, MATCHED_FEATS, MODE_AGNOSTIC, TESTING_MODE
from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import FMRI_BETAS_DIR, IMAGERY_SCENES, model_features_file_path, FUSED_CLS_FEAT_KEY, FUSED_MEAN_FEAT_KEY, \
    DECODER_OUT_DIR
import lightning as pl

MODALITY_SPECIFIC_IMAGES = "images"
MODALITY_SPECIFIC_CAPTIONS = "captions"
MODALITY_AGNOSTIC = "agnostic"

TRAIN_MODE_CHOICES = [MODALITY_AGNOSTIC, MODALITY_SPECIFIC_CAPTIONS, MODALITY_SPECIFIC_IMAGES]

SPLIT_TRAIN = "train"
SPLIT_TEST = "test"
SPLIT_IMAGERY = "imagery"

TEST_BATCH_SIZE = 140


def stim_id_from_beta_file_name(beta_file_name):
    return int(beta_file_name.replace('beta_I', '').replace('beta_C', '').replace(".nii", ''))


def get_fmri_data_paths(betas_dir, subject, mode, split):
    filename_suffix = "*.nii"
    betas_regex = f'betas_{split}*'
    if mode in [MODALITY_SPECIFIC_IMAGES, MODALITY_SPECIFIC_CAPTIONS]:
        betas_regex += f'_{mode[:-1]}'
    fmri_addresses_regex = os.path.join(betas_dir, subject, betas_regex, filename_suffix)

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


def get_fmri_betas_mean_std_path(subject, mode, mask_name=None):
    bold_std_mean_name = f'bold_multimodal_mean_std_{mode}.p'
    if mask_name is not None:
        bold_std_mean_name += f'_mask_{mask_name}'
    return os.path.join(DECODER_OUT_DIR, "normalizations", subject, bold_std_mean_name)


def get_fmri_betas_standardization_transform(subject, training_mode, latent_feats_config):
    std_mean_path = get_fmri_betas_mean_std_path(subject, training_mode)
    if not os.path.isfile(std_mean_path):
        print("Calculating mean and std over whole train set betas for standardization.")
        os.makedirs(os.path.dirname(std_mean_path), exist_ok=True)
        graymatter_mask = load_graymatter_mask(subject)
        latent_features = pickle.load(open(model_features_file_path(latent_feats_config.model_name), 'rb'))
        train_ds = DecodingDataset(subject, training_mode, SPLIT_TRAIN, latent_features, latent_feats_config,
                                   graymatter_mask)
        train_fmri_betas = [beta for beta, _ in tqdm(iter(train_ds), total=len(train_ds))]
        mean_std = {'mean': np.mean(train_fmri_betas, axis=0),
                    'std': np.std(train_fmri_betas, axis=0)}
        pickle.dump(mean_std, open(std_mean_path, 'wb'))

    mean_std = pickle.load(open(std_mean_path, 'rb'))
    return Standardize(mean_std['mean'], mean_std['std'])


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


def get_latents_mean_std_path(subject, latent_feats_config, mode):
    model_std_mean_name = (
        f'{latent_feats_config.model_name}_{latent_feats_config.features}_{latent_feats_config.vision_features}_'
        f'{latent_feats_config.lang_features}_mean_std_{mode}.p')
    return os.path.join(DECODER_OUT_DIR, "normalizations", subject, model_std_mean_name)


def load_latents_transform(subject, latent_feats_config, mode):
    model_std_mean_path = get_latents_mean_std_path(subject, latent_feats_config, mode)
    model_mean_std = pickle.load(open(model_std_mean_path, 'rb'))
    nn_latent_transform = Standardize(model_mean_std['mean'], model_mean_std['std'])

    return nn_latent_transform


def get_latent_feats_standardization_transform(subject, latent_feats_config, training_mode):
    std_mean_path = get_latents_mean_std_path(subject, latent_feats_config, training_mode)
    if not os.path.isfile(std_mean_path):
        print("Calculating mean and std over whole train set latents for standardization.")
        os.makedirs(os.path.dirname(std_mean_path), exist_ok=True)
        graymatter_mask = load_graymatter_mask(subject)
        latent_features = pickle.load(open(model_features_file_path(latent_feats_config.model_name), 'rb'))
        train_ds = DecodingDataset(subject, training_mode, SPLIT_TRAIN, latent_features, latent_feats_config,
                                   graymatter_mask)
        train_latents = np.array([latents for _, latents in tqdm(iter(train_ds), total=len(train_ds))])

        mean_std = {
            'mean': train_latents.mean(axis=0),
            'std': train_latents.std(axis=0),
        }

        pickle.dump(mean_std, open(std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

    nn_latent_transform = load_latents_transform(subject, latent_feats_config, training_mode)

    return nn_latent_transform


class DecodingDataset(Dataset):
    def __init__(self, betas_dir, subject, mode, split, latent_features, latent_feats_config, graymatter_mask,
                 betas_transform=None, latent_feats_transform=None):
        self.data_paths, self.stim_ids, self.stim_types = get_fmri_data_paths(betas_dir, subject, mode, split)
        self.betas_transform = betas_transform
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


def load_graymatter_mask(subject):
    gray_matter_mask_path = get_graymatter_mask_path(subject)
    gray_matter_mask_img = nib.load(gray_matter_mask_path)
    gray_matter_mask_data = gray_matter_mask_img.get_fdata()
    gray_matter_mask = gray_matter_mask_data == 1
    print(f"Gray matter mask size: {gray_matter_mask.sum()}")
    return gray_matter_mask


@dataclass
class LatentFeatsConfig:
    model_name: str
    features: str
    vision_features: str
    lang_features: str


class Standardize:
    def __init__(self, mean, std, eps=1e-8):
        self.mean = mean
        self.std = std + eps  # Avoid division by 0

    def __call__(self, x):
        return (x - self.mean) / self.std


class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, betas_dir, batch_size, subject, training_mode, latent_feats_config, num_workers, cv_split=0,
                 num_cv_splits=5):
        super().__init__()
        assert cv_split < num_cv_splits

        self.betas_dir = betas_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.subject = subject
        self.training_mode = training_mode
        self.latent_feats_config = latent_feats_config

        self.graymatter_mask = load_graymatter_mask(subject)

        self.latents_transform = get_latent_feats_standardization_transform(subject, latent_feats_config,
                                                                            training_mode)

        self.betas_transform = get_fmri_betas_standardization_transform(subject, training_mode, latent_feats_config)

        print("Loading pickle with latent feats..", end=" ")
        latent_features = pickle.load(open(model_features_file_path(latent_feats_config.model_name), 'rb'))
        print("done.")

        self.data = DecodingDataset(
            self.betas_dir,
            self.subject, self.training_mode, SPLIT_TRAIN, latent_features, latent_feats_config, self.graymatter_mask,
            self.betas_transform, self.latents_transform,
        )
        indices = list(range(len(self.data)))
        val_split_size = round(len(self.data) / num_cv_splits)
        val_indices = indices[cv_split * val_split_size: (cv_split + 1) * val_split_size]
        train_indices = [i for i in indices if i not in val_indices]
        self.ds_train = Subset(self.data, train_indices)
        self.ds_val = Subset(self.data, val_indices)
        self.ds_test = DecodingDataset(
            self.betas_dir,
            self.subject, TESTING_MODE, SPLIT_TEST, latent_features, latent_feats_config, self.graymatter_mask,
            self.betas_transform, self.latents_transform,
        )
        self.ds_imagery = DecodingDataset(
            self.betas_dir,
            self.subject, IMAGERY, SPLIT_IMAGERY, latent_features, latent_feats_config, self.graymatter_mask,
            self.betas_transform, self.latents_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=TEST_BATCH_SIZE, num_workers=self.num_workers)
