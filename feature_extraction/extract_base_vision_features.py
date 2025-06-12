import os

import torch
from transformers import AutoFeatureExtractor, ResNetModel, ViTImageProcessor, ViTModel
from PIL import Image

from data import VISION_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY
from feature_extraction.feat_extraction_utils import FeatureExtractor


BATCH_SIZE = 128
SUFFIX = "*bf(1)"


if torch.cuda.is_available():
    device = "cuda"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    device = "cpu"


class ViTFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = self.preprocessor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state

        feats_vision_cls = last_hidden_state[:, 0, :]
        feats_vision_mean = last_hidden_state[:, 1:].mean(axis=1)

        return {
            VISION_MEAN_FEAT_KEY: feats_vision_mean,
            VISION_CLS_FEAT_KEY: feats_vision_cls,
        }


class ResNetFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = self.preprocessor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        feats_vision = outputs.pooler_output.squeeze()

        return {
            VISION_MEAN_FEAT_KEY: feats_vision,
        }


if __name__ == "__main__":
    model_name = 'microsoft/resnet-18'
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ResNetModel.from_pretrained(model_name)
    extractor = ResNetFeatureExtractor(model, feature_extractor, "Resnet-18", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'microsoft/resnet-50'
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ResNetModel.from_pretrained(model_name)
    extractor = ResNetFeatureExtractor(model, feature_extractor, "resnet-50", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'microsoft/resnet-152'
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ResNetModel.from_pretrained(model_name)
    extractor = ResNetFeatureExtractor(model, feature_extractor, "resnet-152", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'microsoft/resnet-152'
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ResNetModel(ResNetModel.from_pretrained(model_name).config)
    extractor = ResNetFeatureExtractor(model, feature_extractor, "Resnet-152-random", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'google/vit-base-patch16-384'
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    extractor = ViTFeatureExtractor(model, feature_extractor, "vit-b-16", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'google/vit-large-patch16-384'
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    extractor = ViTFeatureExtractor(model, feature_extractor, "vit-l-16", BATCH_SIZE, device)
    extractor.extract_features()

    model_name = 'google/vit-huge-patch14-224-in21k'
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTModel.from_pretrained(model_name)
    extractor = ViTFeatureExtractor(model, feature_extractor, "vit-h-14", BATCH_SIZE, device)
    extractor.extract_features()


