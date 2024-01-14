import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from tqdm import tqdm

from transformers import BertTokenizer, VisualBertForPreTraining, VisualBertModel

from feature_extraction.extract_nn_features import COCOSelected
from utils import FEATURES_DIR, CAPTIONS_PATH, COCO_2017_TRAIN_IMAGES_DIR, STIMULI_IDS_PATH, MODEL_FEATURES_FILES

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64

MIN_BOXES = 10
MAX_BOXES = 100


def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    cfg['MODEL']['DEVICE'] = 'cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg


def get_model(cfg):
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model


def prepare_image_inputs(model, cfg, img_list):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

    batched_inputs = [{"image": convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in
                      img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images = ImageList.from_tensors(images, model.backbone.size_divisibility)

    return images, batched_inputs


def get_features(model, images):
    features = model.backbone(images.tensor)
    return features


def get_proposals(model, images, features):
    proposals, _ = model.proposal_generator(images, features)
    return proposals


def get_box_features(model, features, proposals):
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)

    box_features = box_features.reshape(BATCH_SIZE, 1000, 1024)  # depends on your config and batch size
    return box_features, features_list


def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas


def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals):
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    outputs = FastRCNNOutputs(
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    )

    boxes = outputs.predict_boxes()
    scores = outputs.predict_probs()
    image_shapes = outputs.image_shapes

    return boxes, scores, image_shapes


def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes


def select_boxes(cfg, output_boxes, scores):
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach().reshape(1000, 80, 4)
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1] - 1):
        cls_scores = cls_prob[:, cls_ind + 1]
        det_boxes = cls_boxes[:, cls_ind, :]
        keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf


def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes


def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]


MASKRCNN_FEATS_PATH = "data/maskrcnn_feats.p"


def extract_visualbert_features():
    visualbert_model = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, CAPTIONS_PATH, STIMULI_IDS_PATH, 'both')
    dloader = DataLoader(ds, shuffle=False, batch_size=BATCH_SIZE)

    maskrcnn_feats = pickle.load(open(MASKRCNN_FEATS_PATH, "rb"))

    all_feats = dict()
    for ids, captions, img_paths in tqdm(dloader):
        tokens = tokenizer(captions, padding='max_length', max_length=50)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        token_type_ids = torch.tensor(tokens["token_type_ids"])

        visual_embeds = [maskrcnn_feats[id] for id in ids]
        visual_embeds = torch.stack(visual_embeds)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)

        with torch.no_grad():
            outputs = visualbert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                       visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask,
                                       visual_token_type_ids=visual_token_type_ids)

        # average last hidden states over all words:
        last_hidden_states = outputs.last_hidden_state.mean(dim=1)  # TODO correct way?

        for id, feats, path in zip(ids, last_hidden_states, img_paths):
            all_feats[id.item()] = {"multimodal_feature": feats, "image_path": path}

    path_out = MODEL_FEATURES_FILES["VisualBERT"]
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def extract_image_features():
    """Extract image features that will be fed into VisualBERT."""
    ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, CAPTIONS_PATH, STIMULI_IDS_PATH, 'both')
    dloader = DataLoader(ds, shuffle=False, batch_size=BATCH_SIZE)

    cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    cfg = load_config_and_model_weights(cfg_path)

    maskrcnn_model = get_model(cfg)

    all_feats = dict()
    for ids, captions, img_paths in tqdm(dloader):
        with torch.no_grad():
            imgs = [plt.imread(path) for path in img_paths]

            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]

            images, batched_inputs = prepare_image_inputs(maskrcnn_model, cfg, imgs)

            features = get_features(maskrcnn_model, images)

            proposals = get_proposals(maskrcnn_model, images, features)

            box_features, features_list = get_box_features(maskrcnn_model, features, proposals)

            pred_class_logits, pred_proposal_deltas = get_prediction_logits(maskrcnn_model, features_list, proposals)

            boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals)

            output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in
                            range(len(proposals))]

            temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]
            keep_boxes, max_conf = [], []
            for keep_box, mx_conf in temp:
                keep_boxes.append(keep_box)
                max_conf.append(mx_conf)

            keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in
                          zip(keep_boxes, max_conf)]

            visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in
                             zip(box_features, keep_boxes)]

            for id, feats in zip(ids, visual_embeds):
                all_feats[id] = feats

    os.makedirs(os.path.dirname(MASKRCNN_FEATS_PATH), exist_ok=True)
    pickle.dump(all_feats, open(MASKRCNN_FEATS_PATH, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    os.makedirs(FEATURES_DIR, exist_ok=True)
    extract_image_features()
    extract_visualbert_features()