import os

import torch

from transformers import ViltModel

from feature_extraction.feat_extraction_utils import FeatureExtractor
from transformers import ViltProcessor
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

        # Average lang feats while ignoring padding tokens
        mask = inputs.data["attention_mask"]
        mask_expanded = mask.unsqueeze(-1).expand((mask.shape[0], mask.shape[1], language_embeddings.shape[-1]))
        language_embeddings[mask_expanded == 0] = 0
        feats_lang = language_embeddings.sum(axis=1) / mask_expanded.sum(dim=1)

        # Average image features embeddings
        img_embeddings = img_embeddings.mean(dim=1)
        feats_vision_cls = img_embeddings[:, 0, :]
        feats_vision_mean = img_embeddings[:, 1:].mean(axis=1)

        return feats_lang, feats_vision_mean, feats_vision_cls


if __name__ == "__main__":
    # processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    # model = ViltModel(ViltModel.from_pretrained("dandelin/vilt-b32-mlm").config)
    #
    # extractor = ViLTFeatureExtractor(model, processor, "VILT-random", BATCH_SIZE, device)
    # extractor.extract_features()

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

    extractor = ViLTFeatureExtractor(model, processor, "VILT", BATCH_SIZE, device)
    extractor.extract_features()

