import json
import os.path
import pickle

import numpy as np

from feature_extraction.feat_extraction_utils import CoCoDataset
from utils import COCO_IMAGES_DIR, STIM_INFO_PATH, STIMULI_IDS_PATH
from data import LANG_CLS_FEAT_KEY, VISION_MEAN_FEAT_KEY


if __name__ == "__main__":
    stimuli_ids = pickle.load(open(os.path.expanduser("~/data/multimodal_decoding/stimuli_ids.p"), "rb"))

    coco_annotations_val = json.load(
        open(os.path.expanduser("~/data/coco/annotations_trainval2014/annotations/captions_val2014.json")))
    coco_captions_val = dict()
    for ann in coco_annotations_val['annotations']:
        if ann['image_id'] in coco_captions_val:
            coco_captions_val[ann['image_id']].append(ann['caption'])
        else:
            coco_captions_val[ann['image_id']] = [ann['caption']]

    coco_annotations_train = json.load(
        open(os.path.expanduser("~/data/coco/annotations_trainval2014/annotations/captions_train2014.json")))
    coco_captions_train = dict()
    for ann in coco_annotations_train['annotations']:
        if ann['image_id'] in coco_captions_train:
            coco_captions_train[ann['image_id']].append(ann['caption'])
        else:
            coco_captions_train[ann['image_id']] = [ann['caption']]

    coco_captions = coco_captions_train | coco_captions_val

    ds = CoCoDataset(COCO_IMAGES_DIR, STIM_INFO_PATH, STIMULI_IDS_PATH, 'caption')

    caption_indices = []
    for i, idx in enumerate(stimuli_ids):
        found = False
        caption = ds.captions[idx]
        # Corrected captions
        if caption == "A small dog standing inside a car":
            caption_indices.append(i * 5 + 1)
            continue
        elif caption == "a cat sitting in a bathroom sink":
            caption_indices.append(i * 5 + 4)
            continue
        elif caption == "A woman leaning out a window to talk to someone on the sidewal":
            caption_indices.append(i * 5 + 4)
            continue
        elif caption == "a bowl of food in an open microwave":
            caption_indices.append(i * 5 + 1)
            continue
        elif caption == "A bike stands next to a brick wall":
            caption_indices.append(i * 5 + 2)
            continue
        elif caption == "A couple of giraffes standing in some trees":
            caption_indices.append(i * 5 + 4)
            continue
        elif caption == "A group of kids swimming in the ocean around a man on a surfboard":
            caption_indices.append(i * 5 + 3)
            continue
        for j, cap in enumerate(coco_captions[idx]):
            cap1 = cap.replace(".", "").replace(",", "").replace("-", " ").lower().strip()
            cap2 = caption.replace(".", "").replace(",", "").replace("-", " ").lower().strip()
            if cap1 == cap2:
                caption_indices.append(i * 5 + j)
                found = True
                break
        if not found:
            print("not found!!", caption, coco_captions[idx])

    assert len(caption_indices) == len(stimuli_ids)

    feats_l = np.load(os.path.expanduser("~/Downloads/gw_features_tr+cont/t_gw.npy"))[caption_indices]
    feats_v = np.load(os.path.expanduser("~/Downloads/gw_features_tr+cont/v_gw.npy"))[::5]
    all_feats = dict()
    for feat_l, feat_v, id in zip(feats_l, feats_v, stimuli_ids):
        all_feats[id] = {VISION_MEAN_FEAT_KEY: feat_v, LANG_CLS_FEAT_KEY: feat_l}
    path_out = os.path.expanduser("~/data/multimodal_decoding/nn_features/glow-transl-contrastive.p")
    pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    feats_l = np.load(os.path.expanduser("~/Downloads/gw_features_cont/t_gw.npy"))[caption_indices]
    feats_v = np.load(os.path.expanduser("~/Downloads/gw_features_cont/v_gw.npy"))[::5]
    all_feats = dict()
    for feat_l, feat_v, id in zip(feats_l, feats_v, stimuli_ids):
        all_feats[id] = {VISION_MEAN_FEAT_KEY: feat_v, LANG_CLS_FEAT_KEY: feat_l}
    path_out = os.path.expanduser("~/data/multimodal_decoding/nn_features/glow-contrastive.p")
    pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    feats_l = np.load(os.path.expanduser("~/Downloads/gw_features/t_gw.npy"))[caption_indices]
    feats_v = np.load(os.path.expanduser("~/Downloads/gw_features/v_gw.npy"))[::5]
    all_feats = dict()
    for feat_l, feat_v, id in zip(feats_l, feats_v, stimuli_ids):
        all_feats[id] = {VISION_MEAN_FEAT_KEY: feat_v, LANG_CLS_FEAT_KEY: feat_l}
    path_out = os.path.expanduser("~/data/multimodal_decoding/nn_features/glow.p")
    pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    feats_v = np.load(os.path.expanduser("~/Downloads/gw_features/resnet.npy"))[::5]
    all_feats = dict()
    for feat_v, id in zip(feats_v, stimuli_ids):
        all_feats[id] = {VISION_MEAN_FEAT_KEY: feat_v}
    path_out = os.path.expanduser("~/data/multimodal_decoding/nn_features/resnet-50-glow.p")
    pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    feats_l = np.load(os.path.expanduser("~/Downloads/gw_features/bge.npy"))[caption_indices]
    all_feats = dict()
    for feat_l, id in zip(feats_l, stimuli_ids):
        all_feats[id] = {LANG_CLS_FEAT_KEY: feat_l}
    path_out = os.path.expanduser("~/data/multimodal_decoding/nn_features/bge.p")
    pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    all_feats = dict()
    for feat_l, feat_v, id in zip(feats_l, feats_v, stimuli_ids):
        all_feats[id] = {LANG_CLS_FEAT_KEY: feat_l, VISION_MEAN_FEAT_KEY: feat_v}
    path_out = os.path.expanduser("~/data/multimodal_decoding/nn_features/resnet-and-bge.p")
    pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
