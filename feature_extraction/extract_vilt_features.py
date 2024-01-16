import os

import torch

from transformers import ViltModel

from feature_extraction.feat_extraction_utils import FeatureExtractor
from transformers import ViltProcessor
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2


class ViLTFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = processor(images, captions, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state

        text_input_size = inputs.data["input_ids"].shape[1]

        language_embeddings = last_hidden_states[:, :text_input_size]
        img_embeddings = last_hidden_states[:, text_input_size:]

        language_embeddings = language_embeddings.mean(dim=1)   #TODO ignore padding tokens?
        img_embeddings = img_embeddings.mean(dim=1)

        return language_embeddings, img_embeddings


if __name__ == "__main__":
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

    extractor = ViLTFeatureExtractor(model, processor, "VILT", BATCH_SIZE, device)
    extractor.extract_features()

