import os
import pickle

import torch
from tqdm import tqdm

from transformers import FlavaModel, FlavaProcessor

from feature_extraction.feat_extraction_utils import FeatureExtractor
from PIL import Image

from utils import LANG_FEAT_KEY, VISION_FEAT_KEY, MULTIMODAL_FEAT_KEY, model_features_file_path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 10


class FlavaFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = self.preprocessor(
            text=captions, images=images, return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        img_embeddings = outputs.image_embeddings  # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
        language_embeddings = outputs.text_embeddings  # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768
        # Multimodal embeddings can be used for multimodal tasks such as VQA
        multimodal_embeddings = outputs.multimodal_embeddings  # Batch size X (Number of image patches + Text Sequence Length + 3) X Hidden size => 2 X 275 x 768

        att_mask = inputs.data["attention_mask"]
        averaged_lang_embeddings = []
        for emb, mask in zip(language_embeddings, att_mask):
            length_without_padding = mask.sum().item()
            averaged_lang_embeddings.append(emb[:length_without_padding].mean(dim=0))
        language_embeddings = torch.stack(averaged_lang_embeddings)

        # output from CLS tokens
        img_embeddings = img_embeddings[:, 0]
        multimodal_embeddings = multimodal_embeddings[:, 0]

        return language_embeddings, img_embeddings, multimodal_embeddings

    def extract_features(self):
        all_feats = dict()
        for ids, captions, img_paths in tqdm(self.dloader):
            ids = [id.item() for id in ids]
            language_feats_batch, vision_feats_batch, multi_feats_batch = self.extract_features_from_batch(ids, captions, img_paths)
            if language_feats_batch is None:
                language_feats_batch = [None] * len(ids)
            else:
                language_feats_batch = language_feats_batch.cpu().numpy()
            if vision_feats_batch is None:
                vision_feats_batch = [None] * len(ids)
            else:
                vision_feats_batch = vision_feats_batch.cpu().numpy()
            if multi_feats_batch is None:
                multi_feats_batch = [None] * len(ids)
            else:
                multi_feats_batch = multi_feats_batch.cpu().numpy()
            for id, feats_lang, feats_vision, feats_multi in zip(ids, language_feats_batch, vision_feats_batch, multi_feats_batch):
                all_feats[id] = {LANG_FEAT_KEY: feats_lang, VISION_FEAT_KEY: feats_vision, MULTIMODAL_FEAT_KEY: feats_multi}

        path_out = model_features_file_path(self.model_name)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    model_name = "facebook/flava-full"
    processor = FlavaProcessor.from_pretrained(model_name)
    model = FlavaModel.from_pretrained(model_name)

    extractor = FlavaFeatureExtractor(model, processor, "Flava", BATCH_SIZE, device)
    extractor.extract_features()

