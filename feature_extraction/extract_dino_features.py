import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from feature_extraction.feat_extraction_utils import FeatureExtractor
from utils import VISION_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 100


class DinoFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(img_path).convert('RGB') for img_path in img_paths]

        inputs = self.preprocessor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_state = outputs[0]

        feats_vision_cls = last_hidden_state[:, 0, :]
        feats_vision_mean = last_hidden_state[:, 1:].mean(axis=1)

        return {
            VISION_MEAN_FEAT_KEY: feats_vision_mean,
            VISION_CLS_FEAT_KEY: feats_vision_cls,
        }


if __name__ == "__main__":
    model_name = 'facebook/dinov2-base'
    prepocessor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    extractor = DinoFeatureExtractor(model, prepocessor, "dino-base", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'facebook/dinov2-large'
    prepocessor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    extractor = DinoFeatureExtractor(model, prepocessor, "dino-large", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'facebook/dinov2-giant'
    prepocessor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    extractor = DinoFeatureExtractor(model, prepocessor, "dino-giant", BATCH_SIZE, device)
    extractor.extract_features()