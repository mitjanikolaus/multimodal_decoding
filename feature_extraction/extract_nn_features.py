###############################
# Save features of various dnns
# The script contains several utility functions to extract latent vectors (or features) from DNNs and apply PCA if
# necessary. Current functions might not be suitable for all types of models. You can always add your own functions.
# The important thing is to store the data in a python dictionary with the following nested structure (to be compatible
# with other functions and the analysis scripts):
#
# ```
# ---------------
# Language models
# ---------------
#
# dict
# |
# └── coco_stimulus_id (integer): dict
#     |
#     └── 'lingual_feature': 1-d numpy array (feature vector)
#     |
#     └── 'caption': the corresponding caption (string)
#
# -------------
# Vision models
# -------------
#
# dict
# |
# └── coco_stimulus_id (integer): dict
#     |
#     └── 'visual_feature': 1-d numpy array (feature vector)
#     |
#     └── 'name': image file name (string)
###############################

import os
import torch
from transformers import pipeline
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
from os.path import join
from PIL import Image
from torchvision.models import resnet152, vit_l_16, ViT_L_16_Weights, ResNet152_Weights, VisionTransformer
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torchvision.transforms import functional as F, InterpolationMode
from typing import Tuple
from torch import nn, Tensor

from utils import FEATURES_DIR, IMAGES_IMAGERY_CONDITION, COCO_2017_TRAIN_IMAGES_DIR, IMAGE_STIMULI_IDS_PATH, \
    IMAGES_TEST, PCA_NUM_COMPONENTS

NUM_WORKERS = 8

BATCH_SIZE = 512

if torch.cuda.is_available():
    device = "cuda"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = "cpu"

print("device: ", device)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_coco_image(sid, coco_images_dir):
    r"""
    Helper function to find coco image address based on the id.
    """
    file_name = f"{sid:012d}.jpg"
    image_add = list(glob(join(coco_images_dir, '*', file_name)))
    return image_add[0]


class COCOSelected(Dataset):
    r"""
    Pytorch dataset that loads the preselected data from the COCO dataset.
    The preselected data are given in a separate file (`selection_file`).
    """

    def __init__(self, coco_root, selection_file, mode='image', transform=None):
        r"""
        Args:
            `coco_root` (str): address to the coco2017 root folder (= the parent directory of `images` folder)
            `selection_file` (pickle): address to file containing information about the preselected coco entries
            `mode` (str): can be `caption` or `image` to load captions or images, respectively. Default: `image`
            `transform` (callable): data transformation. Default: None
        """
        super().__init__()
        self.data = []
        self.data.extend(IMAGES_IMAGERY_CONDITION)
        self.data.extend(np.load(selection_file, allow_pickle=True))
        self.root = coco_root
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'image':
            img_add = join(self.root, self.data[index][1])
            img = Image.open(img_add).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, self.data[index][0], self.data[index][1]

        elif self.mode == 'caption':
            cap = self.data[index][2]
            if self.transform is not None:
                cap = self.transform(cap)
            return cap, self.data[index][0]


def get_visual_features(model, dataloader):
    r"""
    Helper function to get feature from Pytorch compatible vision models.

    Args:
        `model` (torch.nn): Pytorch model
        `dataloader` (DataLoader): Pytorch data loader to provide the inputs

    Returns:
        `features` (dict): extracted features
    """
    features = {}
    model.to(device)
    model.eval()
    temp = None
    is_vision_transformer = isinstance(model, VisionTransformer)

    def save_rep(layer, inp, output):
        nonlocal temp
        if is_vision_transformer:
            temp = inp[0].cpu().numpy()
        else:
            temp = output.cpu().numpy()

    if is_vision_transformer:
        module = model.heads
    else:
        module = model.avgpool
    module.register_forward_hook(save_rep)

    with tqdm(dataloader, unit="batch") as tepoch:
        for images, ids, names in tepoch:
            with torch.no_grad():
                model(images.to(device))
            for vector, sid, n in zip(temp, ids, names):
                features[int(sid.item())] = {'visual_feature': vector.squeeze(), 'image_name': n}
    return features


def get_lingual_features_transformers(model_name, dataloader):
    r"""
    Helper function to get features from transformer language models available in huggingface.
    The final feature vector will be the average of all tokens.

    Args:
        `model_name` (str): Name of the language model (based on hugging face repositories)
        `dataloader` (DataLoader): Pytorch data loader to provide the inputs

    Returns:
        `features` (dict): extracted features
    """
    features = {}
    model = pipeline('feature-extraction', model=model_name, device=0 if device == "cuda" else device)
    with tqdm(dataloader, unit="batch") as tepoch:
        for captions, ids in tepoch:
            fdata = model(list(captions))
            for vector, sid, caption in zip(fdata, ids, captions):
                v = np.array(vector[0]).mean(axis=0)
                features[int(sid.item())] = {'lingual_feature': v.squeeze(), 'caption': caption}
    return features


def apply_pca(n_components, data_pickle_file):
    r"""
    Applies PCA on the given latent vectors. The function saves the following (pickle) files
    in the same directory as the given pickle file for the latent vectors:

    - PCA module (to be used later to recover original vectors)
    - Scaler module (that used for data normalization before the PCA)
    - Compressed vectors (transformed latent vectors)

    Args:
        `n_components` (int): number of PCs
        `data_pickle_file` (str): address to the pickle file containing latent vectors
    """
    print("Performing PCA..")
    excluded = set(IMAGES_TEST + [a[0] for a in IMAGES_IMAGERY_CONDITION])

    filename = os.path.basename(data_pickle_file)
    dirname = os.path.dirname(data_pickle_file)
    base_filename = filename[:filename.find('.p')]

    with open(data_pickle_file, 'rb') as f:
        data = pickle.load(f)

    # gathering all the stim ids
    stim_ids = list(data.keys())

    # finding keys for feature values
    features_names = [k for k in data[stim_ids[0]].keys() if "feature" in k]

    # normalizing first, then PCA
    transformed_data = {k: {} for k in stim_ids}
    for feature in features_names:
        # creating the combined vector with order of stim_ids
        vectors = []
        all_vectors = []
        for sid in stim_ids:
            if sid not in excluded:  # no influence of the test and imagery sets
                vectors.append(data[sid][feature])

            all_vectors.append(data[sid][feature])

        vectors = np.array(vectors)
        all_vectors = np.array(all_vectors)

        scaler = StandardScaler()
        normalized = scaler.fit_transform(vectors)
        normalized_all_vectors = scaler.transform(all_vectors)

        pca = PCA(n_components=n_components, whiten=True)  # whiten=True: divide by singular values
        pca.fit(normalized)
        reduced_all_vectors = pca.transform(normalized_all_vectors)

        # storing the reduced vectors (keeping stim_id ordering)
        for idx, sid in enumerate(stim_ids):
            transformed_data[sid][feature] = reduced_all_vectors[idx]

        # save PCA and scaler modules for future transformations
        pca_file = join(dirname, f"{base_filename}_pca_{n_components}_pca_{feature}_module.p")
        with open(pca_file, 'wb') as handle:
            pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
        scaler_file = join(dirname, f"{base_filename}_pca_{n_components}_scaler_{feature}_module.p")
        with open(scaler_file, 'wb') as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # filling the rest as before
    for sid in stim_ids:
        for k in data[sid]:
            if k not in transformed_data[sid]:
                transformed_data[sid][k] = data[sid][k]

    # saving transformed data in the same place with PCA_ncomp postfix
    new_file = join(dirname, f"{base_filename}_pca_{n_components}.p")

    with open(new_file, 'wb') as handle:
        pickle.dump(transformed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class ImageClassificationPreprocessing(nn.Module):
    def __init__(
            self,
            *,
            crop_size: int,
            resize_size: int,
            mean: Tuple[float, ...] = IMAGENET_MEAN,
            std: Tuple[float, ...] = IMAGENET_STD,
            interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, max_size=None, antialias=True)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


def extract_visual_features(model, path_out, resize_size, crop_size, interpolation):
    if os.path.isfile(path_out):
        print(f"Skipping feature extraction for {model.__class__.__name__} as output file {path_out} already exists.")
        return
    print(f"Extracting features with {model.__class__.__name__}..")
    preprocessing = ImageClassificationPreprocessing(crop_size=crop_size, resize_size=resize_size,
                                                     interpolation=interpolation)
    ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, IMAGE_STIMULI_IDS_PATH, 'image', transform=preprocessing)
    dloader = DataLoader(ds, shuffle=False, num_workers=0, batch_size=BATCH_SIZE)
    vf = get_visual_features(model, dloader)
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    pickle.dump(vf, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    apply_pca(PCA_NUM_COMPONENTS, path_out)


def extract_linguistic_features(model_name, path_out):
    if os.path.isfile(path_out):
        print(f"Skipping feature extraction for {model_name} as output file {path_out} already exists.")
        return
    print(f"Extracting features with {model.__class__.__name__}..")

    ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, IMAGE_STIMULI_IDS_PATH, 'caption')
    dloader = DataLoader(ds, shuffle=False, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
    cf = get_lingual_features_transformers(model_name, dloader)

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    pickle.dump(cf, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    apply_pca(PCA_NUM_COMPONENTS, path_out)


if __name__ == "__main__":
    ##########
    # ViT-L-16
    ##########
    model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
    model_name = "vit_l_16"
    path_out = f"{FEATURES_DIR}/vit/{model_name}_encoder_selected_coco_crop.p"
    extract_visual_features(model, path_out, resize_size=242, crop_size=224, interpolation=InterpolationMode.BILINEAR)

    ##########
    # ResNet152
    ##########
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
    model_name = "resnet152"
    path_out = f"{FEATURES_DIR}/resnet/{model_name}_avgpool_selected_coco_crop.p"
    extract_visual_features(model, path_out, resize_size=256, crop_size=224, interpolation=InterpolationMode.BILINEAR)

    #########
    # BERT LARGE
    #########
    model_name = 'bert-large-uncased'
    path_out = f"{FEATURES_DIR}/bert/bert_large_avg_selected_coco.p"
    extract_linguistic_features(model_name, path_out)

    #########
    # GPT-2xl
    #########
    model_name = 'gpt2-xl'
    path_out = f"{FEATURES_DIR}/gpt/gpt2_xl_avg_selected_coco.p"
    extract_linguistic_features(model_name, path_out)

# ds = COCOSelected(COCO_2017_IMAGES_DIR, IMAGE_STIMULI_IDS_PATH, 'caption')
# data = np.load(IMAGE_STIMULI_IDS_PATH, allow_pickle=True)
# print(data[0])
# for subject in IMAGERY_SCENES:
#     scenes = IMAGERY_SCENES[subject]
#     for s in scenes:
#         img_add = find_coco_image(s[1], '/home/milad/datasets/coco2017/images')
#         tail = img_add[img_add.find('/images/'):-1].replace('/',"\\\\")
#         print(f"[{s[1]}, '{tail}', '{s[0]}']")
