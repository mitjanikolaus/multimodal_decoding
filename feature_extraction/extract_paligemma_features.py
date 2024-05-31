import os

import torch

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from feature_extraction.feat_extraction_utils import FeatureExtractor
from PIL import Image

from utils import FUSED_MEAN_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

device = "cuda:1" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 10


class PaliGemmaFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = self.preprocessor(
            text=captions, images=images, return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]

        # Average hidden states while ignoring padding tokens
        mask = inputs["attention_mask"]
        mask_expanded = mask.unsqueeze(-1).expand((mask.shape[0], mask.shape[1], last_hidden_states.shape[-1]))
        last_hidden_states[mask_expanded == 0] = 0
        feats_fused_mean = last_hidden_states.mean(dim=1)

        return {
            FUSED_MEAN_FEAT_KEY: feats_fused_mean,
        }


if __name__ == "__main__":
    model_name = "google/paligemma-3b-pt-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    extractor = PaliGemmaFeatureExtractor(model, processor, "paligemma", BATCH_SIZE, device)
    extractor.extract_features()
