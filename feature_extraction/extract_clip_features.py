import os
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from feature_extraction.feat_extraction_utils import FeatureExtractor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

BATCH_SIZE = 128
CLIP_MODEL = "openai/clip-vit-large-patch14"


# def modify_preprocess(preprocess):
#     r"""
#     modifies the default CLIP preprocess and returns two version:
#     1- resize and center crop
#     2- force resize to the target size
#     """
#     transforms_crop = [transforms.Resize(size=preprocess.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True)]
#     transforms_crop.extend(preprocess.transforms[1:])
#
#     return transforms.Compose(transforms_crop)
# #


class CLIPFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(img_path).convert('RGB') for img_path in img_paths]

        inputs = processor(text=captions, images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.text_embeds, outputs.image_embeds


if __name__ == "__main__":
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    extractor = CLIPFeatureExtractor(model, processor, "CLIP", BATCH_SIZE, device)
    extractor.extract_features()
