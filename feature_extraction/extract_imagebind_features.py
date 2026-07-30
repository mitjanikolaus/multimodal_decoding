import os

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from torch.utils.data import DataLoader

from feature_extraction.feat_extraction_utils import FeatureExtractor, CoCoDatasetOriginalCaptions
from data import VISION_CLS_FEAT_KEY, LANG_CLS_FEAT_KEY

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 100


class ImagebindFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(captions, device),
            ModalityType.VISION: data.load_and_transform_vision_data(img_paths, device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return {
            LANG_CLS_FEAT_KEY: embeddings[ModalityType.TEXT],
            VISION_CLS_FEAT_KEY: embeddings[ModalityType.VISION],
        }


class ImagebindFeatureExtractorOriginalCaptions(FeatureExtractor):

    def __init__(self, model, prepocessor=None, model_name=None, batch_size=10, device="cpu", move_model=True, hidden=None):
        super().__init__(
            model, prepocessor, model_name, batch_size, device, move_model, hidden
        )
        self.ds = CoCoDatasetOriginalCaptions()
        self.dloader = DataLoader(self.ds, shuffle=False, batch_size=batch_size)

    def extract_features_from_batch(self, ids, captions, img_paths):
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(captions, device),
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return {
            LANG_CLS_FEAT_KEY: embeddings[ModalityType.TEXT],
        }


if __name__ == "__main__":
    model = imagebind_model.imagebind_huge(pretrained=True)
    extractor = ImagebindFeatureExtractorOriginalCaptions(model, None, "imagebind_original_captions", BATCH_SIZE, device)
    extractor.extract_features()

    # model = imagebind_model.imagebind_huge(pretrained=True)
    # extractor = ImagebindFeatureExtractor(model, None, "imagebind", BATCH_SIZE, device)
    # extractor.extract_features()
    #
    # model = imagebind_model.imagebind_huge(pretrained=False)
    # extractor = ImagebindFeatureExtractor(model, None, "random-imagebind", BATCH_SIZE, device)
    # extractor.extract_features()
