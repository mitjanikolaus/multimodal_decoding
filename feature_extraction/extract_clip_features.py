import os
import torch
import clip
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from glob import glob
from os.path import join as opj
from PIL import Image
import pickle
from tqdm import tqdm

from feature_extraction.extract_nn_features import COCOSelected, apply_pca, PCA_NUM_COMPONENTS

from utils import FEATURES_DIR, IMAGES_IMAGERY_CONDITION, IMAGE_STIMULI_IDS_PATH, COCO_2017_TRAIN_IMAGES_DIR

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

BATCH_SIZE = 128
CLIP_MODELS = ['ViT-L/14@336px']


def modify_preprocess(preprocess):
    r"""
    modifies the default CLIP preprocess and returns two version:
    1- resize and center crop
    2- force resize to the target size
    """
    transforms_crop = [transforms.Resize(size=preprocess.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True)]
    transforms_crop.extend(preprocess.transforms[1:])

    return transforms.Compose(transforms_crop)


def load_images(folder_address, extension="jpg"):
    f"""
    loads all images as PIL
    """
    
    adds   = glob(opj(folder_address, f"*{extension}"))
    names  = [os.path.basename(f) for f in adds]
    images = []
    for idx, f in enumerate(adds):
        if idx % 100 == 0:
            print(f"{idx + 1}/{len(adds)}", end='\r')
        images.append(Image.open(f))
    return images, names


def extract_visual_features():
    for clip_model in CLIP_MODELS:
        print("Extracting visual features with ", clip_model)
        model, preprocess = clip.load(clip_model, device=device, jit=False)
        preprocess_crop = modify_preprocess(preprocess)

        ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, IMAGE_STIMULI_IDS_PATH, 'image', transform=preprocess_crop)
        dataloader = DataLoader(ds, shuffle=False, num_workers=0, batch_size=BATCH_SIZE)

        all_feats = dict()
        model.eval()
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, ids, names in tepoch:
                with torch.no_grad():
                    feats_batch = model.encode_image(images.to(device)).cpu().numpy()
                    for id, feats, name in zip(ids, feats_batch, names):
                        all_feats[id] = {"visual_features": feats, "image_name": name}

        path_out = os.path.join(FEATURES_DIR, "clip", "clip_v_VITL14336px_selected_coco_dataset_crop.p")

        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def extract_language_features():
    r"""
    Since there are several captions and typos in the COCO dataset we will use the chosen and edited ones to
    be consistent with the experiment.
    """
    for clip_model in CLIP_MODELS:
        print("Extracting language features with ", clip_model)

        images_stimuli = np.load(IMAGE_STIMULI_IDS_PATH, allow_pickle=True)

        for data in [IMAGES_IMAGERY_CONDITION, images_stimuli]:
            captions = [c[2] for c in data]
            stim_ids = [c[0] for c in data]
            model, preprocess = clip.load(clip_model, device=device, jit=False)
            model.eval()

            all_feats = dict()
            print("extracting linguistic feats.. ")
            for idx in tqdm(range(0, len(captions), BATCH_SIZE)):
                end = min(idx + BATCH_SIZE, len(captions))

                captions_batch = captions[idx:end]
                stim_ids_batch = stim_ids[idx:end]

                with torch.no_grad():
                    text = clip.tokenize(captions_batch).to(device)
                    text_features = model.encode_text(text).cpu().numpy()

                for i, _ in enumerate(range(idx, end)):
                    if stim_ids_batch[i] in all_feats:
                        raise Exception('Key already exists: ', stim_ids_batch[i])
                    id = stim_ids_batch[i]
                    temp = dict()
                    temp['lingual_feature'] = text_features[i]
                    temp['caption'] = captions_batch[i]
                    all_feats[id] = temp
            print("done.")
            print(len(all_feats))

            if data == IMAGES_IMAGERY_CONDITION:
                path_out = os.path.join(FEATURES_DIR, "clip", "clip_l_VITL14336px_imagery_coco_dataset_crop.p")
            else:
                path_out = os.path.join(FEATURES_DIR, "clip", "clip_l_VITL14336px_selected_coco_dataset_crop.p")

            pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    extract_visual_features()
    extract_language_features()

    apply_pca(PCA_NUM_COMPONENTS, f"{FEATURES_DIR}/clip/clip_l_VITL14336px_selected_coco_dataset_crop.p")
    apply_pca(PCA_NUM_COMPONENTS, f"{FEATURES_DIR}/clip/clip_v_VITL14336px_selected_coco_dataset_crop.p")
