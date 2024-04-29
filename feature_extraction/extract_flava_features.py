import os
import pickle

import torch
from torch import nn
from tqdm import tqdm

from transformers import FlavaModel, FlavaProcessor

from feature_extraction.feat_extraction_utils import FeatureExtractor
from PIL import Image

from utils import LANG_FEAT_KEY, VISION_MEAN_FEAT_KEY, VISION_CLS_FEAT_KEY, MULTIMODAL_FEAT_KEY, \
    model_features_file_path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 100


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

        last_hidden_states = outputs.multimodal_embeddings

        img_input_size = outputs.image_embeddings.shape[1]

        language_embeddings = last_hidden_states[:, img_input_size + 1:]    # (+1 for multimodal CLS token)
        image_embeddings = last_hidden_states[:, 1:img_input_size + 1]    # (+1 for multimodal CLS token)

        # Average lang feats while ignoring padding tokens
        mask = inputs.data["attention_mask"]
        mask_expanded = mask.unsqueeze(-1).expand((mask.shape[0], mask.shape[1], language_embeddings.shape[-1]))
        language_embeddings[mask_expanded == 0] = 0
        feats_lang = language_embeddings.sum(axis=1) / mask_expanded.sum(dim=1)

        feats_vision_cls = image_embeddings[:, 0, :]
        feats_vision_mean = image_embeddings[:, 1:].mean(axis=1)

            # text_embeddings = outputs.text_embeddings
            # image_embeddings = outputs.image_embeddings
            #
            # text_embedding = model.text_projection(text_embeddings[:, 0, :])
            # text_embedding = nn.functional.normalize(text_embedding, dim=-1)
            #
            # image_embedding = model.image_projection(image_embeddings[:, 0, :])
            # image_embedding = nn.functional.normalize(image_embedding, dim=-1)

        feats_multi = last_hidden_states.mean(axis=1)

        return feats_lang, feats_vision_mean, feats_vision_cls, feats_multi

    def extract_features(self):
        all_feats = dict()
        for ids, captions, img_paths in tqdm(self.dloader):
            ids = [id.item() for id in ids]
            language_feats_batch, vision_mean_feats_batch, vision_cls_feats_batch, multi_feats_batch = self.extract_features_from_batch(
                ids, captions, img_paths)
            if language_feats_batch is None:
                language_feats_batch = [None] * len(ids)
            else:
                language_feats_batch = language_feats_batch.cpu().numpy()
            if vision_mean_feats_batch is None:
                vision_mean_feats_batch = [None] * len(ids)
            else:
                vision_mean_feats_batch = vision_mean_feats_batch.cpu().numpy()
            if vision_cls_feats_batch is None:
                vision_cls_feats_batch = [None] * len(ids)
            else:
                vision_cls_feats_batch = vision_cls_feats_batch.cpu().numpy()
            if multi_feats_batch is None:
                multi_feats_batch = [None] * len(ids)
            else:
                multi_feats_batch = multi_feats_batch.cpu().numpy()
            for id, feats_lang, feats_vision_mean, feats_vision_cls, feats_multi in zip(ids,
                                                                                        language_feats_batch,
                                                                                        vision_mean_feats_batch,
                                                                                        vision_cls_feats_batch,
                                                                                        multi_feats_batch):
                all_feats[id] = {LANG_FEAT_KEY: feats_lang, VISION_MEAN_FEAT_KEY: feats_vision_mean,
                                 VISION_CLS_FEAT_KEY: feats_vision_cls, MULTIMODAL_FEAT_KEY: feats_multi}

        path_out = model_features_file_path(self.model_name)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    model_name = "facebook/flava-full"
    processor = FlavaProcessor.from_pretrained(model_name)
    model = FlavaModel.from_pretrained(model_name)

    extractor = FlavaFeatureExtractor(model, processor, "flava-alt", BATCH_SIZE, device)
    extractor.extract_features()

    # processor = FlavaProcessor.from_pretrained(model_name)
    # model = FlavaModel(FlavaModel.from_pretrained(model_name).config)
    # model.init_weights()
    #
    # extractor = FlavaFeatureExtractor(model, processor, "random-flava", BATCH_SIZE, device)
    # extractor.extract_features()
