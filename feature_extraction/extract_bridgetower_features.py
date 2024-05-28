import os

import torch

from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning

from feature_extraction.feat_extraction_utils import FeatureExtractor
from PIL import Image

from utils import LANG_FEAT_KEY, VISION_CLS_FEAT_KEY, FUSED_CLS_FEAT_KEY, FUSED_MEAN_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

device = "cuda:1" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32


class BridgeTowerFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = processor(images, captions, padding=True, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # img_embeddings = outputs.image_embeds
        # language_embeddings = outputs.text_embeds

        hidden_states_multi = outputs.hidden_states[2]
        last_hidden_states_multi = hidden_states_multi[-1]
        feats_fused_mean = torch.cat(last_hidden_states_multi, dim=1).mean(dim=1)

        return {
            # LANG_FEAT_KEY: language_embeddings,   # features are not language-only, but fused!
            # VISION_CLS_FEAT_KEY: img_embeddings,   # features are not vision-only, but fused!
            FUSED_MEAN_FEAT_KEY: feats_fused_mean,
            FUSED_CLS_FEAT_KEY: outputs.cross_embeds
        }


if __name__ == "__main__":
    model_name = "BridgeTower/bridgetower-large-itm-mlm-itc"
    processor = BridgeTowerProcessor.from_pretrained(model_name)
    model = BridgeTowerForContrastiveLearning.from_pretrained(model_name)

    extractor = BridgeTowerFeatureExtractor(model, processor, "bridgetower-large", BATCH_SIZE, device)
    extractor.extract_features()





