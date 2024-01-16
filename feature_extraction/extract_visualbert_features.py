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
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures.boxes import Boxes
from detectron2.layers import nms, ShapeSpec
from detectron2 import model_zoo
from detectron2.config import get_cfg
from tqdm import tqdm

from transformers import BertTokenizer, VisualBertModel

from feature_extraction.feat_extraction_utils import FeatureExtractor, COCOSelected
from utils import CAPTIONS_PATH, COCO_2017_TRAIN_IMAGES_DIR, STIMULI_IDS_PATH

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 5

MIN_BOXES = 10
MAX_BOXES = 100

BOX_FEATURES_DIM = 1024
MASKRCNN_FEATS_PATH = "data/maskrcnn_feats.p"


def load_config_and_model_weights(cfg_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    cfg['MODEL']['DEVICE'] = device

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
    model = model.to(device)
    return model


def prepare_image_inputs(model, cfg, img_list):
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.tensor(x.astype("float32").transpose(2, 0, 1)).to(device)

    batched_inputs = [{"image": convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in
                      img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN, device=device).view(num_channels, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD, device=device).view(num_channels, 1, 1)
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

    box_features_per_sample = []
    for proposal in proposals:
        num_proposals = len(proposal.proposal_boxes)
        box_features_per_sample.append(box_features[:num_proposals])
        box_features = box_features[num_proposals:]

    return box_features_per_sample, features_list


def get_prediction_logits(model, features_list, proposals):
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas


def get_box_scores(output_layers, pred_class_logits, pred_proposal_deltas, proposals):
    boxes = output_layers.predict_boxes((pred_class_logits, pred_proposal_deltas), proposals)
    scores = output_layers.predict_probs((pred_class_logits, pred_proposal_deltas), proposals)

    return boxes, scores


def get_output_boxes(boxes, batched_inputs, image_size):
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes


def calc_max_confs(cfg, output_boxes, scores):
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach().reshape(-1, 80, 4)
    max_conf = torch.zeros((cls_boxes.shape[0]), device=device)
    for cls_ind in range(0, cls_prob.shape[1] - 1):
        cls_scores = cls_prob[:, cls_ind + 1]
        det_boxes = cls_boxes[:, cls_ind, :]
        keep = nms(det_boxes, cls_scores, test_nms_thresh).cpu().numpy()
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    return max_conf


def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf.cpu()).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf.cpu()).numpy()[::-1][:max_boxes]
    return keep_boxes


def get_visual_embeds(box_features, keep_boxes):
    return box_features[keep_boxes.copy()]


def extract_image_features():
    """Extract image features that will be fed into VisualBERT."""
    ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, CAPTIONS_PATH, STIMULI_IDS_PATH, 'both')
    dloader = DataLoader(ds, shuffle=False, batch_size=BATCH_SIZE)

    cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    cfg = load_config_and_model_weights(cfg_path)

    maskrcnn_model = get_model(cfg)
    output_layers = FastRCNNOutputLayers(
        **FastRCNNOutputLayers.from_config(cfg=cfg, input_shape=ShapeSpec(channels=BOX_FEATURES_DIM)))

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

            boxes, scores = get_box_scores(output_layers, pred_class_logits, pred_proposal_deltas, proposals)

            output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in
                            range(len(proposals))]

            max_confs = [calc_max_confs(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]

            keep_boxes = [torch.where(max_conf >= cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)[0] for max_conf in max_confs]

            keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in
                          zip(keep_boxes, max_confs)]

            visual_embeds = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in
                             zip(box_features, keep_boxes)]

            for id, feats in zip(ids, visual_embeds):
                all_feats[id.item()] = feats.cpu().numpy()

    os.makedirs(os.path.dirname(MASKRCNN_FEATS_PATH), exist_ok=True)
    pickle.dump(all_feats, open(MASKRCNN_FEATS_PATH, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


class VisualBERTFeatureExtractor(FeatureExtractor):

    def __init__(self, model, prepocessor=None, model_name=None, batch_size=10, device="cpu"):
        super(VisualBERTFeatureExtractor, self).__init__(model, prepocessor, model_name, batch_size, device)
        self.maskrcnn_feats = pickle.load(open(MASKRCNN_FEATS_PATH, "rb"))

    def extract_features_from_batch(self, ids, captions, img_paths):
        tokens = self.preprocessor(captions, padding='max_length', max_length=50)
        input_ids = torch.tensor(tokens["input_ids"], device=device)
        attention_mask = torch.tensor(tokens["attention_mask"], device=device)
        token_type_ids = torch.tensor(tokens["token_type_ids"], device=device)

        visual_embeds = [torch.tensor(self.maskrcnn_feats[id], device=device) for id in ids]
        visual_embeds = torch.stack(visual_embeds)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)
        visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask,
                                       visual_token_type_ids=visual_token_type_ids)

        # average last hidden states over all words:
        last_hidden_states = outputs.last_hidden_state.mean(dim=1)  # TODO correct way?

        # language and vision features are the same (they were internally merged)
        return last_hidden_states, last_hidden_states


if __name__ == "__main__":
    # extract_image_features()

    model = VisualBertModel.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    extractor = VisualBERTFeatureExtractor(model, tokenizer, "VisualBERT", BATCH_SIZE, device)
    extractor.extract_features()
