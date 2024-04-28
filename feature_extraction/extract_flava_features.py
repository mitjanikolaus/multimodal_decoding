import os

import torch
from torch import nn

from transformers import FlavaModel, FlavaProcessor

from feature_extraction.feat_extraction_utils import FeatureExtractor
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda:1" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 10


class FlavaFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        print("captions: ", captions)
        inputs = self.preprocessor(
            text=captions, images=images, return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        text_embeddings = outputs.text_embeddings
        image_embeddings = outputs.image_embeddings

        print("text embeddings: ", text_embeddings.shape)
        print("image embeddings: ", image_embeddings.shape)

        text_embedding = model.text_projection(text_embeddings[:, 0, :])
        text_embedding = nn.functional.normalize(text_embedding, dim=-1)

        image_embedding = model.image_projection(image_embeddings[:, 0, :])
        image_embedding = nn.functional.normalize(image_embedding, dim=-1)

        feats_vision_mean = image_embeddings[:, 1:].mean(axis=1)

        print("text_embedding: ", text_embedding.shape)
        print("image_embedding: ", image_embedding.shape)

        return text_embedding, feats_vision_mean, image_embedding


if __name__ == "__main__":
    model_name = "facebook/flava-full"
    processor = FlavaProcessor.from_pretrained(model_name)
    model = FlavaModel.from_pretrained(model_name)

    extractor = FlavaFeatureExtractor(model, processor, "flava", BATCH_SIZE, device)
    extractor.extract_features()

    # processor = FlavaProcessor.from_pretrained(model_name)
    # model = FlavaModel(FlavaModel.from_pretrained(model_name).config)
    # model.init_weights()
    #
    # extractor = FlavaFeatureExtractor(model, processor, "random-flava", BATCH_SIZE, device)
    # extractor.extract_features()



