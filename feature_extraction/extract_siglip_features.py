import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor

from feature_extraction.feat_extraction_utils import FeatureExtractor
from data import LANG_CLS_FEAT_KEY, VISION_CLS_FEAT_KEY, VISION_MEAN_FEAT_KEY, LANG_MEAN_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128


class SigLipFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(img_path).convert('RGB') for img_path in img_paths]

        inputs = processor(text=captions, images=images, padding="max_length", return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        text_mean_feats = outputs.text_model_output.last_hidden_state.mean(dim=1)
        vision_mean_feats = outputs.vision_model_output.last_hidden_state.mean(dim=1)

        print(f"text_mean_feats shape: {text_mean_feats.shape}")
        print(f"vision_mean_feats shape: {vision_mean_feats.shape}")

        return {
            LANG_CLS_FEAT_KEY: outputs.text_embeds,
            VISION_CLS_FEAT_KEY: outputs.image_embeds,
            LANG_MEAN_FEAT_KEY: text_mean_feats,
            VISION_MEAN_FEAT_KEY: vision_mean_feats,
        }


if __name__ == "__main__":
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

    extractor = SigLipFeatureExtractor(model, processor, "siglip", BATCH_SIZE, device)
    extractor.extract_features()
