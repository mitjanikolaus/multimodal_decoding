import os
from PIL import Image
from transformers import AutoModel, BitsAndBytesConfig
from transformers import CLIPImageProcessor, CLIPTokenizer
import torch

from feature_extraction.feat_extraction_utils import FeatureExtractor
from data import LANG_CLS_FEAT_KEY, VISION_CLS_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128


class EVACLIPFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(img_path).convert('RGB') for img_path in img_paths]

        input_ids = self.preprocessor.tokenizer(captions, return_tensors="pt", padding=True).input_ids.to(torch.float8).to(device)
        input_pixels = self.preprocessor(images=images, return_tensors="pt", padding=True).pixel_values.to(torch.float8).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(input_pixels)
            text_features = self.model.encode_text(input_ids)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return {
            LANG_CLS_FEAT_KEY: text_features,
            VISION_CLS_FEAT_KEY: image_features,
        }


if __name__ == "__main__":
    model_name_or_path = "BAAI/EVA-CLIP-8B"
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )

    processor.tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
    extractor = EVACLIPFeatureExtractor(model, processor, "eva-clip", BATCH_SIZE, device, move_model=False)
    extractor.extract_features()
