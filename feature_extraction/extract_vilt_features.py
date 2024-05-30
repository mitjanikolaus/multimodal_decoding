import os

import torch

from transformers import ViltModel

from feature_extraction.feat_extraction_utils import FeatureExtractor
from transformers import ViltProcessor
from PIL import Image

from utils import FUSED_MEAN_FEAT_KEY, FUSED_CLS_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

device = "cuda:1" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 10


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
        language_embeddings_masked = language_embeddings.clone()
        language_embeddings_masked[mask_expanded == 0] = 0

        feats_fused_mean = (language_embeddings_masked.sum(axis=1) + img_embeddings[:, 1:].sum(axis=1)) / (
                    mask_expanded.sum(dim=1) + img_embeddings[:, 1:].shape[1])
        feats_fused_cls = outputs.pooler_output

        return {
            FUSED_MEAN_FEAT_KEY: feats_fused_mean,
            FUSED_CLS_FEAT_KEY: feats_fused_cls,
        }


if __name__ == "__main__":
    # processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    # model = ViltModel(ViltModel.from_pretrained("dandelin/vilt-b32-mlm").config)
    #
    # extractor = ViLTFeatureExtractor(model, processor, "VILT-random", BATCH_SIZE, device)
    # extractor.extract_features()

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

    extractor = ViLTFeatureExtractor(model, processor, "vilt", BATCH_SIZE, device)
    extractor.extract_features()
