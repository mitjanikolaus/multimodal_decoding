import os
import pickle

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from tqdm import tqdm

from feature_extraction.feat_extraction_utils import FeatureExtractor
from utils import MODEL_FEATURES_FILES


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 10


class ImagebindFeatureExtractor(FeatureExtractor):

    def extract_features(self):
        all_feats = dict()
        all_feats_avg = dict()
        for ids, captions, img_paths in tqdm(self.dloader):
            feats_batch_img, feats_batch_text = self.extract_features_from_batch(ids, captions, img_paths)

            for id, feats_img, feats_text, path in zip(ids, feats_batch_img, feats_batch_text, img_paths):
                feats_concatenated = torch.cat((feats_img, feats_text)).cpu().numpy()
                all_feats[id.item()] = {"multimodal_feature": feats_concatenated, "image_path": path}

                feats_avg = torch.stack((feats_img, feats_text)).mean(dim=0).cpu().numpy()
                all_feats_avg[id.item()] = {"multimodal_feature": feats_avg, "image_path": path}

        path_out = MODEL_FEATURES_FILES[self.model_name]
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        path_out = MODEL_FEATURES_FILES[f"{self.model_name}_AVG"]
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats_avg, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def extract_features_from_batch(self, ids, captions, img_paths):
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(captions, device),
            ModalityType.VISION: data.load_and_transform_vision_data(img_paths, device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return embeddings[ModalityType.VISION], embeddings[ModalityType.TEXT]


if __name__ == "__main__":
    model = imagebind_model.imagebind_huge(pretrained=True)
    extractor = ImagebindFeatureExtractor(model, None, "ImageBind", BATCH_SIZE, device)
    extractor.extract_features()
