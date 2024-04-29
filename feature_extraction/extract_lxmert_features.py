import os
import torch
from tqdm import tqdm
from transformers import LxmertModel, LxmertTokenizer

from feature_extraction.feat_extraction_utils import FeatureExtractor
import base64
import numpy as np
import csv
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1 # currently only working with batch size 1 because bottom up feats box numbers are not aligned

BOTTOM_UP_FEATS_FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

BOTTOM_UP_TRAIN_0_FEATS_PATH = os.path.expanduser(
    "~/data/coco/bottom_up_feats/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0")
BOTTOM_UP_TRAIN_1_FEATS_PATH = os.path.expanduser(
    "~/data/coco/bottom_up_feats/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1")
BOTTOM_UP_VAL_FEATS_PATH = os.path.expanduser(
    "~/data/coco/bottom_up_feats/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv")
BOTTOM_UP_TEST_FEATS_PATH = os.path.expanduser(
    "~/data/coco/bottom_up_feats/trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv")


def load_bottom_up_features():
    csv.field_size_limit(sys.maxsize)
    print("reading bottom up features from tsv..")

    def read_tsv(file_path):
        data = {}
        with open(file_path, "r") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=BOTTOM_UP_FEATS_FIELDNAMES)
            for item in tqdm(reader):
                item['image_id'] = int(item['image_id'])
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item['num_boxes'] = int(item['num_boxes'])
                for field in ['boxes', 'features']:
                    item[field] = np.frombuffer(base64.decodebytes(bytes(item[field], encoding="utf8")),
                                                dtype=np.float32).reshape((item['num_boxes'], -1))
                data[item['image_id']] = item
        return data

    feats_test = read_tsv(BOTTOM_UP_TEST_FEATS_PATH)
    feats_val = read_tsv(BOTTOM_UP_VAL_FEATS_PATH)
    feats_train_0 = read_tsv(BOTTOM_UP_TRAIN_0_FEATS_PATH)
    feats_train_1 = read_tsv(BOTTOM_UP_TRAIN_1_FEATS_PATH)
    return feats_val | feats_test | feats_train_0 | feats_train_1


class LXMERTFeatureExtractor(FeatureExtractor):

    def __init__(self, model, prepocessor=None, model_name=None, batch_size=10, device="cpu"):
        super(LXMERTFeatureExtractor, self).__init__(model, prepocessor, model_name, batch_size, device)
        self.bottom_up_img_features = load_bottom_up_features()

    def transform_box_feats(self, embeddings):
        obj_num = embeddings['num_boxes']
        boxes = embeddings['boxes'].copy()
        assert obj_num == len(boxes)

        img_h, img_w = embeddings['image_h'], embeddings['image_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        return torch.tensor(boxes, device=self.device)

    def extract_features_from_batch(self, ids, captions, img_paths):
        visual_embeds = [self.bottom_up_img_features[id] for id in ids]

        visual_feats = torch.stack([torch.tensor(embeds['features'], device=self.device) for embeds in visual_embeds])
        visual_pos = torch.stack([self.transform_box_feats(embeds) for embeds in visual_embeds])

        visual_attention_mask = torch.ones(visual_feats.shape[:-1], dtype=torch.float)

        inputs = self.preprocessor(captions, return_tensors="pt")
        inputs['visual_attention_mask'] = visual_attention_mask
        inputs['visual_feats'] = visual_feats
        inputs['visual_pos'] = visual_pos
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        feats_lang = outputs.pooled_output
        feats_vision = outputs.vision_output.mean(dim=1)
        feats_vision_cls = outputs.vision_output[0]
        return feats_lang, feats_vision, feats_vision_cls


if __name__ == "__main__":
    checkpoint = "unc-nlp/lxmert-base-uncased"
    tokenizer = LxmertTokenizer.from_pretrained(checkpoint)
    model = LxmertModel.from_pretrained(checkpoint)

    extractor = LXMERTFeatureExtractor(model, tokenizer, "lxmert", BATCH_SIZE, device)
    extractor.extract_features()
