import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from feature_extraction.feat_extraction_utils import FeatureExtractor


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 10


class DynoFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(img_path).convert('RGB') for img_path in img_paths]

        inputs = self.preprocessor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs[0]

        feats_vision = last_hidden_states[:, 0, :]
        return None, feats_vision


if __name__ == "__main__":
    model_name = 'facebook/dinov2-base'
    prepocessor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    extractor = DynoFeatureExtractor(model, prepocessor, "dino-base", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'facebook/dinov2-large'
    prepocessor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    extractor = DynoFeatureExtractor(model, prepocessor, "dino-large", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'facebook/dinov2-giant'
    prepocessor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    extractor = DynoFeatureExtractor(model, prepocessor, "dino-giant", BATCH_SIZE, device)
    extractor.extract_features()