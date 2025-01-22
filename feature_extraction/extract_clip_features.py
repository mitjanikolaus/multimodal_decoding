import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from feature_extraction.feat_extraction_utils import FeatureExtractor
from data import LANG_CLS_FEAT_KEY, VISION_CLS_FEAT_KEY, LANG_MEAN_FEAT_KEY, VISION_MEAN_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

device = "cuda:0" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
CLIP_MODEL = "openai/clip-vit-large-patch14"


class CLIPFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(img_path).convert('RGB') for img_path in img_paths]

        inputs = processor(text=captions, images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        return {
            LANG_CLS_FEAT_KEY: outputs.text_embeds,
            LANG_MEAN_FEAT_KEY: outputs.text_model_output.last_hidden_state.mean(dim=1),
            VISION_CLS_FEAT_KEY: outputs.image_embeds,
            VISION_MEAN_FEAT_KEY: outputs.vision_model_output.last_hidden_state.mean(dim=1),
        }


if __name__ == "__main__":
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    extractor = CLIPFeatureExtractor(model, processor, "clip", BATCH_SIZE, device)
    extractor.extract_features()
