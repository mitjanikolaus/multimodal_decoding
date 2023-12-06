###############################
# Save features of various dnns
# The script contains several utility functions to extract latent vectors (or features) from DNNs and apply PCA if
# necessary. Current functions might not be suitable for all types of models. You can always add your own functions.
# The importatnt thing is to store the data in a python dictionary with the following nested structure (to be compatible
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
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from transformers import pipeline
import numpy as np
from   torch.utils.data import DataLoader, Dataset
from   glob import glob
from   os.path import join as opj
from   PIL import Image
from torchvision.models import resnet152, vit_l_16
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from decoding_utils import IMAGERY_SCENES
from torchvision.transforms import functional as F, InterpolationMode
from typing import Tuple
from torch import nn, Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

backback = "\\"
imagery_samples = [
    [406591, f'images{backback}train2017{backback}000000406591.jpg', 'A woman sits in a beach chair as a man walks along the sand'],
    [324670, f'images{backback}train2017{backback}000000324670.jpg', 'White bird sitting in front of a lighthouse with a red roof'],
    [563723, f'images{backback}train2017{backback}000000563723.jpg', 'A little cat standing on the lap of a man sitting in a chair'],
    [406591, f'images{backback}train2017{backback}000000406591.jpg', 'A woman sits in a beach chair as a man walks along the sand'],
    [563723, f'images{backback}train2017{backback}000000563723.jpg', 'A little cat standing on the lap of a man sitting in a chair'],
    [254568, f'images{backback}train2017{backback}000000254568.jpg', 'A lonely giraffe is walking in the middle of a grassy field'],
    [42685,  f'images{backback}train2017{backback}000000042685.jpg', 'A party of young people in a bedroom with a large box of pizza'],
    [473773, f'images{backback}train2017{backback}000000473773.jpg', 'A man walking across a field of snow holding skis and ski poles'],
    [485909, f'images{backback}train2017{backback}000000485909.jpg', 'Two men are discussing something next to a desk with a laptop'],
    [544502, f'images{backback}train2017{backback}000000544502.jpg', 'A young male holding a racket and tennis ball in a tennis court'],
    [556512, f'images{backback}train2017{backback}000000556512.jpg', 'A group of kids swimming in the ocean around a man on a surfboard'],
    [410573, f'images{backback}train2017{backback}000000410573.jpg', 'A cat and a dog staring intensely at each other on an armchair'],
    [48670,  f'images{backback}train2017{backback}000000048670.jpg', 'A man stands by a rainy street with an umbrella over his head'],
    [263212, f'images{backback}train2017{backback}000000263212.jpg', 'A woman working on her computer while also using her cell phone'],
    [214816, f'images{backback}train2017{backback}000000214816.jpg', 'An old broken down church and graffiti on surrounding structures'],
    [141343, f'images{backback}train2017{backback}000000141343.jpg', 'A teddy bear lying down on the sidewalk in front of a store'],
    [213506, f'images{backback}train2017{backback}000000213506.jpg', 'A woman leaning out a window to talk to someone on the sidewal'],
    [162396, f'images{backback}train2017{backback}000000162396.jpg', 'The man on the skateboard and the dog are getting their picture taken'],
]

test_stim_ids = [3862,
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
    image_add = list(glob(opj(coco_images_dir, '*', file_name)))
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
        self.data.extend(imagery_samples)
        self.data.extend(np.load(selection_file, allow_pickle=True))
        self.root = coco_root
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'image':
            img_add = opj(self.root, self.data[index][1].replace('\\','/'))
            img = Image.open(img_add).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, self.data[index][0], self.data[index][1]
        
        elif self.mode == 'caption':
            cap = self.data[index][2]
            if self.transform is not None:
                cap = self.transform(cap)
            return cap, self.data[index][0]


def get_visual_features(model, module, save_input, dataloader):
    r"""
    Helper function to get feature from Pytorch compatible vision models.

    Args:
        `model` (torch.nn): Pytorch model
        `module` (torch.nn): Features will be extracted from this module
        `save_input` (boolean): If `True`, module's input will be saved instead of its output
        `dataloader` (DataLoader): Pytorch data loader to provide the inputs

    Returns:
        `features` (dict): extracted feautures
    """
    features = {}
    model.to(device)
    model.eval()
    temp = None
    def save_rep(layer, inp, output):
        nonlocal temp
        if save_input:
            temp = inp[0].cpu().numpy()
        else:
            temp = output.cpu().numpy()
    module.register_forward_hook(save_rep)

    with tqdm(dataloader, unit="batch") as tepoch:
        for images, ids, names in tepoch:
            with torch.no_grad():
                model(images.to(device))
            # print(temp[0].shape)
            for vector, sid, n in zip(temp, ids, names):
                features[int(sid.item())] = {'visual_feature': vector.squeeze(), 'image_name': n}
    return features


def get_lingual_features_transformers(model_name, dataloader):
    r"""
    Helper function to get feature from transformer language models available in hugging face.
    The final feature vector will be average of all tokens.

    Args:
        `model_name` (str): Name of the language model (based on hugging face repositories)
        `dataloader` (DataLoader): Pytorch data loader to provide the inputs

    Returns:
        `features` (dict): extracted feautures
    """
    features = {}
    model = pipeline('feature-extraction', model=model_name, device=0)
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
    forbidden = set(test_stim_ids + [a[0] for a in imagery_samples])

    filename = os.path.basename(data_pickle_file)
    dirname  = os.path.dirname(data_pickle_file)
    pickle_index = filename.find('.pickle')

    with open(data_pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # gathering all the stim ids
    stim_ids = list(data.keys())

    # finding keys for feature values
    features = []
    for k in data[stim_ids[0]]:
        if 'feature' in k:
            features.append(k)
    
    # normalizing first, then PCA
    transformed_data = {k:{} for k in stim_ids}
    for feature in features:
        # creating the combined vector with order of stim_ids
        vectors = []
        all_vectors = []
        for sid in stim_ids:
            if sid not in forbidden:    # no influence of the test and imagery sets
                vectors.append(data[sid][feature])
            
            all_vectors.append(data[sid][feature])

        vectors = np.array(vectors)
        all_vectors = np.array(all_vectors)

        scaler = StandardScaler()
        normal_vectors = scaler.fit_transform(vectors)
        normal_all_vectors = scaler.transform(all_vectors)

        pca = PCA(n_components=n_components, whiten=True)   # whiten=True: devide by singular values
        pca.fit(normal_vectors)
        reduced_all_vectors = pca.transform(normal_all_vectors)

        # storing the reduced vectors (keeping stim_id ordering)
        for idx, sid in enumerate(stim_ids):
            transformed_data[sid][feature] = reduced_all_vectors[idx]
        
        pca_file = opj(dirname, f"{filename[:pickle_index]}_pca_{n_components}_pca_{feature}_module.pickle")
        with open(pca_file, 'wb') as handle:
            pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
        scaler_file = opj(dirname, f"{filename[:pickle_index]}_pca_{n_components}_scaler_{feature}_module.pickle")
        with open(scaler_file, 'wb') as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # filling the rest as before
    for sid in stim_ids:
        for k in data[sid]:
            if k not in transformed_data[sid]:
                transformed_data[sid][k] = data[sid][k]

    # saving in the same place with PCA_ncomp postfix
    filename = os.path.basename(data_pickle_file)
    dirname  = os.path.dirname(data_pickle_file)
    pickle_index = filename.find('.pickle')
    new_file = opj(dirname, f"{filename[:pickle_index]}_pca_{n_components}.pickle")

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


class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
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

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# preprocessing = transforms.Compose([
#     transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
#     # transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize,
# ])

##########
# ViT-L-16
##########
# preprocessing = ImageClassification(crop_size=224, resize_size=224)
# ds = COCOSelected('/home/milad/datasets/coco2017/', '/mnt/HD1/milad/cocoproject/ShortCaptions_AllStims_CorrectedTyposInTestCaptions', 'image', transform=preprocessing)
# print(len(ds))
# vitl16 = vit_l_16(True)
# dloader = DataLoader(ds, shuffle=False, num_workers=0, batch_size=256)
# vf = get_visual_features(vitl16, vitl16.heads, True, dloader)
# # print(vitl16)
# with open(f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/vit/vit_l_16_encoder_selected_coco_crop.pickle", 'wb') as handle:
#     pickle.dump(vf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# apply_pca(768, "/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/vit/vit_l_16_encoder_selected_coco_crop.pickle")

##########
#ResNet152
##########
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# preprocessing = transforms.Compose([
#     transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
#     # transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     normalize,
# ])
# ds = COCOSelected('/home/milad/datasets/coco2017/', '/mnt/HD1/milad/cocoproject/ShortCaptions_AllStims_CorrectedTyposInTestCaptions', 'image', transform=preprocessing)
# print(len(ds))
# dloader = DataLoader(ds, shuffle=False, num_workers=8, batch_size=256)
# resnet = resnet152(True)
# vf = get_visual_features(resnet, resnet.avgpool, False, dloader)
# with open(f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/resnet/resnet152_avgpool_selected_coco_crop.pickle", 'wb') as handle:
#     pickle.dump(vf, handle, protocol=pickle.HIGHEST_PROTOCOL)
# apply_pca(768, f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/resnet/resnet152_avgpool_selected_coco_crop.pickle")

#########
# GPT-2xl
#########
# ds = COCOSelected('/home/milad/datasets/coco2017/', '/mnt/HD1/milad/cocoproject/ShortCaptions_AllStims_CorrectedTyposInTestCaptions', 'caption')
# dloader = DataLoader(ds, shuffle=False, num_workers=8, batch_size=256)
# cf = get_lingual_features_transformers('gpt2-xl', dloader)
# with open(f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/gpt/gpt2_xl_avg_selected_coco.pickle", 'wb') as handle:
#     pickle.dump(cf, handle, protocol=pickle.HIGHEST_PROTOCOL)
# apply_pca(768, f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/gpt/gpt2_xl_avg_selected_coco.pickle")

#########
# BERT LARGE
#########
ds = COCOSelected('/home/milad/datasets/coco2017/', '/mnt/HD1/milad/cocoproject/ShortCaptions_AllStims_CorrectedTyposInTestCaptions', 'caption')
dloader = DataLoader(ds, shuffle=False, num_workers=8, batch_size=256)
cf = get_lingual_features_transformers('bert-large-uncased', dloader)
with open(f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/bert/bert_large_avg_selected_coco.pickle", 'wb') as handle:
    pickle.dump(cf, handle, protocol=pickle.HIGHEST_PROTOCOL)
apply_pca(768, f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/bert/bert_large_avg_selected_coco.pickle")

# apply_pca(768, '/home/milad/projects/multimodal_decoding/latent_vectors/resnet152/avgpool_imagery_coco_crop.pickle')
# combine_visual_and_lingual_features_in_one_file(
#     '/home/milad/projects/multimodal_decoding/latent_vectors/resnet152/avgpool_selected_coco_crop_PCA_768.pickle',
#     '/home/milad/projects/multimodal_decoding/latent_vectors/gpt2_xl/avg_selected_coco_PCA_768.pickle',
#     '/home/milad/projects/multimodal_decoding/latent_vectors/resnet_gpt/resnet_avgpool_gpt2xl_avg_selected_coco_PCA_768.pickle'
# )
# apply_pca(768, '/mnt/HD1/milad/multimodal_decoding/latent_vectors/gpt/gpt2_xl_avg_selected_coco.pickle')

# combine_visual_and_lingual_features_in_one_file('/mnt/HD1/milad/multimodal_decoding/latent_vectors/vit/vit_l_16_encoder_selected_coco_crop_v.pickle',
# '/mnt/HD1/milad/multimodal_decoding/latent_vectors/gpt/gpt2_xl_avg_selected_coco.pickle',
# '/mnt/HD1/milad/multimodal_decoding/latent_vectors/vitl16_gpt2xl/vitl16_encoder_gpt2xl_avg_selected_coco_crop_vl.pickle')

# combine_visual_and_lingual_features_in_one_file('/mnt/HD1/milad/multimodal_decoding/latent_vectors/resnet/resnet152_avgpool_selected_coco_crop_v_pca_768.pickle',
# '/mnt/HD1/milad/multimodal_decoding/latent_vectors/gpt/gpt2_xl_avg_selected_coco_pca_768.pickle',
# '/mnt/HD1/milad/multimodal_decoding/latent_vectors/resnet152xgptxl/resnet152_avgpool_gpt2xl_avg_selected_coco_crop_vl.pickle')

# combine_visual_and_lingual_features_in_one_file('/mnt/HD1/milad/multimodal_decoding/latent_vectors/resnet/resnet152_avgpool_selected_coco_crop_v.pickle',
# '/mnt/HD1/milad/multimodal_decoding/latent_vectors/gpt/gpt2_xl_avg_selected_coco.pickle',
# '/mnt/HD1/milad/multimodal_decoding/latent_vectors/resnet152xgptxl/resnet152_avgpool_gpt2xl_avg_selected_coco_crop_vl.pickle')


# apply_pca(768, '/mnt/HD1/milad/multimodal_decoding/latent_vectors/vitl16_gpt2xl/vitl16_encoder_gpt2xl_avg_selected_coco_crop_vl.pickle')

# ds = COCOSelected('/home/milad/datasets/coco2017/', '/mnt/HD1/milad/cocoproject/ShortCaptions_AllStims_CorrectedTyposInTestCaptions', 'caption')
# data = np.load('/mnt/HD1/milad/cocoproject/ShortCaptions_AllStims_CorrectedTyposInTestCaptions', allow_pickle=True)
# print(data[0])
# for subject in IMAGERY_SCENES:
#     scenes = IMAGERY_SCENES[subject]
#     for s in scenes:
#         img_add = find_coco_image(s[1], '/home/milad/datasets/coco2017/images')
#         tail = img_add[img_add.find('/images/'):-1].replace('/',"\\\\")
#         print(f"[{s[1]}, '{tail}', '{s[0]}']")

# apply_pca(768, f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/clip/clip_l_VITL14336px_selected_coco_dataset_crop.pickle")
# apply_pca(768, f"/mnt/HD1/milad/multimodal_decoding/latent_vectors_uni_modal/clip/clip_v_VITL14336px_selected_coco_dataset_crop.pickle")
