import os

import torch

from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from feature_extraction.feat_extraction_utils import FeatureExtractor
from PIL import Image

from data import FUSED_MEAN_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

BATCH_SIZE = 1


class PaliGemmaFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = self.preprocessor(
            text=captions, images=images, return_tensors="pt",
        )
        # print(f'input ids : {inputs["input_ids"]}')
        # print(f'input ids shape: {inputs["input_ids"].shape}')
        # print(f'pixel_values shape: {inputs["pixel_values"].shape}')

        # mask = inputs["attention_mask"]
        # print(f"mask:\n {mask}")
        # print(f"inputs: {inputs}")

        inputs = inputs.to(torch.bfloat16).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]

        # Average hidden states while ignoring padding tokens
        # mask_expanded = mask.unsqueeze(-1).expand((mask.shape[0], mask.shape[1], last_hidden_states.shape[-1]))
        # print(f"last_hidden_states shape {last_hidden_states.shape}")
        # print(f"mask expanded shape {mask_expanded.shape}")
        # print(f"mask_expanded[0]:\n {torch.sum(mask_expanded[0])}")
        # print(f"mask_expanded[1]:\n {torch.sum(mask_expanded[1])}")
        # print(f"mask_expanded[3]:\n {torch.sum(mask_expanded[3])}")
        # print(f"last_hidden_states[0]:\n {last_hidden_states[0]}")
        # print(f"last_hidden_states[1]:\n {last_hidden_states[1]}")
        # print(f"last_hidden_states[3]:\n {last_hidden_states[3]}")
        # last_hidden_states[mask_expanded == 0] = 0

        feats_fused_mean = last_hidden_states.mean(dim=1)
        print(f"feats_fused_mean shape {feats_fused_mean.shape}")
        print(f"feats_fused_mean {feats_fused_mean}")

        return {
            FUSED_MEAN_FEAT_KEY: feats_fused_mean,
        }


if __name__ == "__main__":
    # model_name = "google/paligemma-3b-pt-224"
    # model = PaliGemmaForConditionalGeneration.from_pretrained(model_name).eval()
    # processor = PaliGemmaProcessor.from_pretrained(model_name)
    #
    # extractor = PaliGemmaFeatureExtractor(model, processor, "paligemma", BATCH_SIZE, device)
    # extractor.extract_features()

    model_name = "google/paligemma2-3b-pt-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    processor = PaliGemmaProcessor.from_pretrained(model_name)

    extractor = PaliGemmaFeatureExtractor(model, processor, "paligemma2", BATCH_SIZE, device, move_model=False)
    extractor.extract_features()
