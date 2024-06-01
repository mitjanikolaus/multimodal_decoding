import os

import torch
from lavis.models import load_model_and_preprocess


from feature_extraction.feat_extraction_utils import FeatureExtractor
from PIL import Image

from utils import LANG_FEAT_KEY, VISION_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY, FUSED_CLS_FEAT_KEY, FUSED_MEAN_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

device = "cuda:1" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 100


class PerceiverFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        vis_processors, txt_processors = self.preprocessor
        image_input = torch.stack([vis_processors["eval"](image) for image in images]).to(device)
        text_input = [txt_processors["eval"](caption) for caption in captions]
        sample = {"image": image_input, "text_input": text_input}

        with torch.no_grad():
            features_multimodal = model.extract_features(sample)
            feats_fused_mean = features_multimodal.multimodal_embeds.mean(dim=1)
            feats_fused_cls = features_multimodal.multimodal_embeds[:, 0, :]

            features_image = model.extract_features(sample, mode="image")
            feats_vision_mean = features_image.image_embeds_proj.mean(dim=1)
            feats_vision_cls = features_image.image_embeds_proj[:, 0, :]

            features_text = model.extract_features(sample, mode="text")
            feats_lang = features_text.text_embeds_proj.mean(dim=1)

        return {
            LANG_FEAT_KEY: feats_lang,
            VISION_MEAN_FEAT_KEY: feats_vision_mean,
            VISION_CLS_FEAT_KEY: feats_vision_cls,
            FUSED_MEAN_FEAT_KEY: feats_fused_mean,
            FUSED_CLS_FEAT_KEY: feats_fused_cls
        }


if __name__ == "__main__":
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
                                                                      model_type="pretrain", is_eval=True,
                                                                      device=device)

    processors = (vis_processors, txt_processors)

    extractor = PerceiverFeatureExtractor(model, prepocessor=processors, model_name="blip2-alt", batch_size=BATCH_SIZE,
                                          device=device)
    extractor.extract_features()
