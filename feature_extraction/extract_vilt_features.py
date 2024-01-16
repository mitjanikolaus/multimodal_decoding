import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import ViltModel

from feature_extraction.extract_nn_features import COCOSelected
from feature_extraction.feat_extraction_utils import FeatureExtractor
from utils import FEATURES_DIR, CAPTIONS_PATH, COCO_2017_TRAIN_IMAGES_DIR, STIMULI_IDS_PATH, MODEL_FEATURES_FILES
from transformers import ViltProcessor
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2


class ViLTFeatureExtractor(FeatureExtractor):

    def extract_features(self):
        all_feats = dict()
        all_feats_avg = dict()
        for ids, captions, img_paths in tqdm(self.dloader):
            text_embeddings, img_embeddings, embeddings = self.extract_features_from_batch(ids, captions, img_paths)
            for id, feats_avg, path, text_embedding, img_embedding in zip(ids, embeddings, img_paths,
                                                                          text_embeddings, img_embeddings):
                concatenated = torch.cat((text_embedding, img_embedding))
                all_feats[id.item()] = {"multimodal_feature": concatenated.cpu().numpy(), "image_path": path}
                all_feats_avg[id.item()] = {"multimodal_feature": feats_avg.cpu().numpy(), "image_path": path}

        path_out = MODEL_FEATURES_FILES["VILT"]
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        path_out = MODEL_FEATURES_FILES["VILT_AVG"]
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats_avg, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = processor(images, captions, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state

        text_input_size = inputs.data["input_ids"].shape[1]

        text_embeddings = last_hidden_states[:, :text_input_size]
        img_embeddings = last_hidden_states[:, text_input_size:]

        text_embeddings = text_embeddings.mean(dim=1)   #TODO ignore padding tokens?
        img_embeddings = img_embeddings.mean(dim=1)

        general_embeddings = last_hidden_states.mean(dim=1)

        return text_embeddings, img_embeddings, general_embeddings


if __name__ == "__main__":
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

    extractor = ViLTFeatureExtractor(model, processor, "VILT", BATCH_SIZE, device)
    extractor.extract_features()

