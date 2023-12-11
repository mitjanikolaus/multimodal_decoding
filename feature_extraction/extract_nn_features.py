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
from torchvision import transforms
from transformers import pipeline
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
from os.path import join
from PIL import Image
from torchvision.models import resnet152, vit_l_16, ViT_L_16_Weights, ResNet152_Weights, VisionTransformer, ResNet
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from decoding_utils import IMAGERY_SCENES
from torchvision.transforms import functional as F, InterpolationMode
from typing import Tuple
from torch import nn, Tensor

NUM_WORKERS = 8

BATCH_SIZE = 512

if torch.cuda.is_available():
    device = "cuda"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = "cpu"

print("device: ", device)

DATA_DIR = os.path.expanduser("~/data/multimodal_decoding")
FEATURES_DIR = os.path.join(DATA_DIR, "feats_unimodal")
os.makedirs(FEATURES_DIR, exist_ok=True)

COCO_2017_TRAIN_IMAGES_DIR = os.path.expanduser("~/data/coco/images/train_2017")

IMAGE_STIMULI_IDS_PATH = os.path.join(DATA_DIR, "ShortCaptions_AllStims_CorrectedTyposInTestCaptionsLinux.p")

PCA_NUM_COMPONENTS = 768

CROP_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

IMAGES_TEST = [3862,
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
               573980]


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
        # TODO
        self.data = self.data[:100]
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
    base_filename = filename[:filename.find('.pickle')]

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
        pca_file = join(dirname, f"{base_filename}_pca_{n_components}_pca_{feature}_module.pickle")
        with open(pca_file, 'wb') as handle:
            pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
        scaler_file = join(dirname, f"{base_filename}_pca_{n_components}_scaler_{feature}_module.pickle")
        with open(scaler_file, 'wb') as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # filling the rest as before
    for sid in stim_ids:
        for k in data[sid]:
            if k not in transformed_data[sid]:
                transformed_data[sid][k] = data[sid][k]

    # saving transformed data in the same place with PCA_ncomp postfix
    new_file = join(dirname, f"{base_filename}_pca_{n_components}.pickle")

    with open(new_file, 'wb') as handle:
        pickle.dump(transformed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_visual_and_lingual_features_in_one_file(visual_pickle_file, lingual_pickle_file, output_pickle_file):
    r"""
    Combines linguistic and visual pickle files into a single file.
    """
    with open(visual_pickle_file, 'rb') as f:
        data_v = pickle.load(f)

    with open(lingual_pickle_file, 'rb') as ff:
        data_l = pickle.load(ff)

    print('number of samples:', len(data_v), len(data_l))
    # gathering all the stim ids
    stim_ids = list(data_v.keys())
    combined_data = {}
    for sid in stim_ids:
        visual = data_v[sid]
        lingual = data_l[sid]
        combined_data[sid] = {**visual, **lingual}

    with open(output_pickle_file, 'wb') as handle:
        pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class ImageClassificationPreprocessing(nn.Module):
    def __init__(
            self,
            *,
            crop_size: int,
            resize_size: int = 256,
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


def extract_visual_features(model, path_out):
    if os.path.isfile(path_out):
        print(f"Skipping feature extraction for {model.__class__.__name__} as output file {path_out} already exists.")
        return
    print(f"Extracting features with {model.__class__.__name__}..")
    preprocessing = ImageClassificationPreprocessing(crop_size=CROP_SIZE, resize_size=CROP_SIZE)
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
    model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
    model_name = "vit_l_16"
    path_out = f"{FEATURES_DIR}/vit/{model_name}_encoder_selected_coco_crop.pickle"
    extract_visual_features(model, path_out)

    ##########
    # ResNet152
    ##########
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model_name = "resnet152"
    path_out = f"{FEATURES_DIR}/resnet/{model_name}_avgpool_selected_coco_crop.pickle"
    extract_visual_features(model, path_out)

    #########
    # BERT LARGE
    #########
    model_name = 'bert-large-uncased'
    path_out = f"{FEATURES_DIR}/bert/bert_large_avg_selected_coco.pickle"
    extract_linguistic_features(model_name, path_out)

    #########
    # GPT-2xl
    #########
    model_name = 'gpt2-xl'
    path_out = f"{FEATURES_DIR}/gpt/gpt2_xl_avg_selected_coco.pickle"
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
