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

from feature_extraction.extract_nn_features import IMAGE_STIMULI_IDS_PATH, COCO_2017_TRAIN_IMAGES_DIR, \
    COCOSelected, FEATURES_DIR, IMAGES_IMAGERY_CONDITION, apply_pca, PCA_NUM_COMPONENTS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

BATCH_SIZE = 128


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


CLIP_MODELS = ['ViT-L/14@336px']    # TODO, 'RN50x64']


def save_visual_features():
    for clip_model in CLIP_MODELS:
        print("Extracting visual features with ", clip_model)
        model, preprocess = clip.load(clip_model, device=device, jit=False)
        preprocess_crop = modify_preprocess(preprocess)

        ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, IMAGE_STIMULI_IDS_PATH, 'image', transform=preprocess_crop)
        dataloader = DataLoader(ds, shuffle=False, num_workers=0, batch_size=BATCH_SIZE)

        model.eval()
        with tqdm(dataloader, unit="batch") as tepoch:
            for images, ids, names in tepoch:
                image_features = []

                with torch.no_grad():
                    image_features.append(model.encode_image(images.to(device)).cpu().numpy())

            image_features = np.concatenate(image_features, axis=0)
            print(image_features.shape)
            path_out = os.path.join(FEATURES_DIR, "clip", f"{clip_model.replace('/', '').replace('@', '-')}_train_dataset_crop.npz")
            os.makedirs(os.path.dirname(path_out), exist_ok=True)
            np.savez(path_out, visual_features=image_features, image_names=names)


def add_language_features_to_visuals_with_selected_captions():
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
            path_names = [c[1] for c in data]
            model, preprocess = clip.load(clip_model, device=device, jit=False)
            model.eval()

            language_feats = dict()
            clip_captions = dict()
            clip_path_names = dict()
            print("extracting linguistic feats.. ", end="")
            for idx in tqdm(range(0, len(captions), BATCH_SIZE)):
                print(f"{idx}/{len(captions)}", end="\r")
                end = min(idx + BATCH_SIZE, len(captions))

                captions_batch = captions[idx:end]
                stim_ids_batch = stim_ids[idx:end]
                stim_names_batch = path_names[idx:end]

                with torch.no_grad():
                    text = clip.tokenize(captions_batch).to(device)
                    text_features = model.encode_text(text).cpu().numpy()

                for i, _ in enumerate(range(idx, end)):
                    if stim_ids_batch[i] in language_feats:
                        raise Exception('Key already exists: ', stim_ids_batch[i])
                    id = stim_ids_batch[i]
                    language_feats[id] = text_features[i]
                    clip_captions[id] = captions_batch[i]
                    clip_path_names[id] = stim_names_batch[i]
            print("done.")
            print(len(language_feats))

            # Load visual feature file
            path_vis_feats = os.path.join(FEATURES_DIR, "clip", f"{clip_model.replace('/', '').replace('@', '-')}_train_dataset_crop.npz")

            print(f"loading visual feats from " + path_vis_feats)
            feats = np.load(path_vis_feats)

            clip_reps = dict()
            coco_visual_features = feats['visual_features']
            coco_image_names = feats['image_names']
            coco_stim_ids = [int(os.path.basename(name)[:-4]) for name in coco_image_names]
            print("merging..") # TODO: no!!!
            for coco_idx in range(len(coco_visual_features)):
                if coco_stim_ids[coco_idx] in language_feats:
                    temp = dict()
                    temp['visual_feature'] = coco_visual_features[coco_idx]
                    temp['image_name'] = coco_image_names[coco_idx]
                    if os.path.basename(temp['image_name']) != os.path.basename(clip_path_names[coco_stim_ids[coco_idx]]):
                        a = temp['image_name']
                        b = os.path.basename(clip_path_names[coco_stim_ids[coco_idx]])
                        raise Exception(f"Name mismatch: {a} and {b}")
                    temp['image_name'] = clip_path_names[coco_stim_ids[coco_idx]]
                    temp['lingual_feature'] = language_feats[coco_stim_ids[coco_idx]]
                    temp['caption'] = clip_captions[coco_stim_ids[coco_idx]]
                    clip_reps[coco_stim_ids[coco_idx]] = temp

            print(len(clip_reps))
            if data == IMAGES_IMAGERY_CONDITION:
                path_out = os.path.join(FEATURES_DIR, "clip", f"{clip_model.replace('/', '').replace('@', '-')}_imagery_coco_dataset_crop_vl.pickle")
            else:
                path_out = os.path.join(FEATURES_DIR, "clip", f"{clip_model.replace('/', '').replace('@', '-')}_selected_coco_dataset_crop_vl.pickle")
            os.makedirs(os.path.dirname(path_out), exist_ok=True)

            with open(path_out, 'wb') as handle:
                pickle.dump(clip_reps, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
        #     # Create an extra array for visual features and fill it based on the file names and the previous dictionary
        #     lingual_features = []
        #     lingual_captions = []
        #     for i in range(len(visual_names)):
        #         if i % 1000 == 0:
        #             print(f"{i}/{len(visual_names)}", end="\r")
        #         lingual_features.append(fname2feature[visual_names[i]])
        #         lingual_captions.append(fname2caption[visual_names[i]])
        #     lingual_features = np.array(lingual_features)
        #     print()
        #     print(lingual_features.shape)

        #     np.savez(f"{clip_model.replace('/','').replace('@', '-')}_{js_name}_dataset_crop_vl.npy",
        #             visual_features=visual_file['features'], image_names=visual_file['names'], lingual_features=lingual_features, captions=lingual_captions)


if __name__ == "__main__":
    # save_visual_features()
    add_language_features_to_visuals_with_selected_captions()

    ################
    # separate clip-l and clip-v
    ################
    path = os.path.join(FEATURES_DIR, "clip", "ViT-L14-336px_selected_coco_dataset_crop_vl.pickle")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(data[list(data.keys())[0]].keys())
    dict_v = {d: {'visual_feature':data[d]['visual_feature'], 'image_name': data[d]['image_name']} for d in data}
    dict_l = {d: {'lingual_feature':data[d]['lingual_feature'], 'caption': data[d]['caption']} for d in data}
    print(dict_v[list(dict_v.keys())[0]].keys())
    print(dict_l[list(dict_l.keys())[0]].keys())
    path_out_visual = os.path.join(FEATURES_DIR, "clip", "clip_v_VITL14336px_selected_coco_dataset_crop.pickle")
    with open(path_out_visual, 'wb') as handle:
        pickle.dump(dict_v, handle, protocol=pickle.HIGHEST_PROTOCOL)

    path_out_linguistic = os.path.join(FEATURES_DIR, "clip", "clip_l_VITL14336px_selected_coco_dataset_crop.pickle")
    with open(path_out_linguistic, 'wb') as handle:
        pickle.dump(dict_l, handle, protocol=pickle.HIGHEST_PROTOCOL)

    apply_pca(PCA_NUM_COMPONENTS, f"{FEATURES_DIR}/clip/clip_l_VITL14336px_selected_coco_dataset_crop.pickle")
    apply_pca(PCA_NUM_COMPONENTS, f"{FEATURES_DIR}/clip/clip_v_VITL14336px_selected_coco_dataset_crop.pickle")
