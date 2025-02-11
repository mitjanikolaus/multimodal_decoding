import os

import numpy as np
import torch

from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig, BatchFeature
from transformers.models.paligemma.processing_paligemma import IMAGE_TOKEN, build_string_from_input

from feature_extraction.feat_extraction_utils import FeatureExtractor
from PIL import Image

from data import FUSED_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY, VISION_MEAN_FEAT_KEY, LANG_CLS_FEAT_KEY, LANG_MEAN_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 10


class PaliGemmaFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        # inputs_image_only = self.preprocessor(
        #     text=[IMAGE_TOKEN for _ in images], images=images, return_tensors="pt",
        # )
        #
        # inputs_image_only = inputs_image_only.to(torch.float16).to(device)
        # with torch.no_grad():
        #     outputs = self.model(**inputs_image_only, output_hidden_states=True)
        #
        # last_hidden_states = outputs.hidden_states[-1]
        #
        # vision_feats_cls = last_hidden_states[:, 0]
        # vision_feats_mean_alt = outputs.image_hidden_states.mean(dim=1)
        # vision_feats_mean = last_hidden_states.mean(dim=1)
        #
        # input_strings = [
        #     build_string_from_input(
        #         prompt=caption,
        #         bos_token=processor.tokenizer.bos_token,
        #         image_seq_len=processor.image_seq_length,
        #         image_token=IMAGE_TOKEN,
        #         num_images=0
        #     )
        #     for caption in captions
        # ]
        # return_data = processor.tokenizer(
        #     input_strings,
        #     return_token_type_ids=False,
        #     return_tensors="pt",
        #     padding=True,
        # )
        # inputs_text_only = BatchFeature(data=return_data)
        #
        # mask_text_only = inputs_text_only["attention_mask"]
        #
        # inputs_text_only = inputs_text_only.to(torch.float16).to(device)
        # with torch.no_grad():
        #     outputs = self.model(**inputs_text_only, output_hidden_states=True)
        #
        # last_hidden_states = outputs.hidden_states[-1]
        #
        # # Average hidden states while ignoring padding tokens
        # mask_text_only_expanded = mask_text_only.unsqueeze(-1).expand(
        #     (mask_text_only.shape[0], mask_text_only.shape[1], last_hidden_states.shape[-1]))
        # last_hidden_states[mask_text_only_expanded == 0] = 0
        #
        # lang_feats_mean = last_hidden_states.mean(dim=1)

        inputs = self.preprocessor(
            text=captions, images=images, return_tensors="pt", padding=True,
        )
        input_ids = inputs["input_ids"]

        inputs = inputs.to(torch.float16).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]

        # Average hidden states separately for img and text feats, then average across modalities
        feats_fused_mean = []
        for last_hidden_state, inputs in zip(last_hidden_states, input_ids):
            # we need to find the first img token idx to ignore the padding tokens that are prepended
            first_img_token_index = (inputs == self.preprocessor.image_token_id).nonzero(as_tuple=True)[0][0]
            first_img_token_index = first_img_token_index.cpu().item()

            bos_index = ((inputs == self.preprocessor.tokenizer.bos_token_id).nonzero(as_tuple=True)[0])
            bos_index = bos_index.cpu().item()

            img_feats = last_hidden_state[first_img_token_index:bos_index]
            lang_feats = last_hidden_state[bos_index:]

            fused = torch.mean(torch.vstack((torch.mean(img_feats, dim=0), torch.mean(lang_feats, dim=0))), dim=0)
            feats_fused_mean.append(fused)

        feats_fused_mean = torch.stack(feats_fused_mean)
        # feats_fused_mean = last_hidden_states.mean(dim=1)

        return {
            # LANG_MEAN_FEAT_KEY: lang_feats_mean,
            # VISION_MEAN_FEAT_KEY: vision_feats_mean,
            # "vision_features_mean_alt": vision_feats_mean_alt,
            # VISION_CLS_FEAT_KEY: vision_feats_cls,
            FUSED_MEAN_FEAT_KEY: feats_fused_mean,
        }


if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_name = "google/paligemma2-3b-pt-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name, quantization_config=quantization_config, device_map="cuda"
    )
    processor = PaliGemmaProcessor.from_pretrained(model_name)

    extractor = PaliGemmaFeatureExtractor(model, processor, "paligemma2_new_fused", BATCH_SIZE, device,
                                          move_model=False)
    extractor.extract_features()

    model_name = "google/paligemma-3b-pt-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name, quantization_config=quantization_config, device_map="cuda"
    )
    processor = PaliGemmaProcessor.from_pretrained(model_name)

    extractor = PaliGemmaFeatureExtractor(model, processor, "paligemma_new_fused", BATCH_SIZE, move_model=False)
    extractor.extract_features()
