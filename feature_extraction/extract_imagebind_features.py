import os
import pickle

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import COCOSelected
from utils import FEATURES_DIR, CAPTIONS_PATH, COCO_2017_TRAIN_IMAGES_DIR, STIMULI_IDS_PATH, MODEL_FEATURES_FILES


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 10


def extract_features():
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, CAPTIONS_PATH, STIMULI_IDS_PATH, 'both')
    dloader = DataLoader(ds, shuffle=False, batch_size=BATCH_SIZE)

    all_feats = dict()
    all_feats_avg = dict()
    for ids, captions, img_paths in tqdm(dloader):

        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(captions, device),
            ModalityType.VISION: data.load_and_transform_vision_data(img_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        # print(
        #     "Vision x Text: ",
        #     torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
        # )

        for id, feats_img, feats_text, path in zip(ids, embeddings[ModalityType.VISION], embeddings[ModalityType.TEXT], img_paths):
            feats_concatenated = torch.cat((feats_img, feats_text))
            all_feats[id.item()] = {"multimodal_feature": feats_concatenated.cpu().numpy(), "image_path": path}

            feats_avg = torch.stack((feats_img, feats_text)).mean(dim=0)
            all_feats_avg[id.item()] = {"multimodal_feature": feats_avg.cpu().numpy(), "image_path": path}

    path_out = MODEL_FEATURES_FILES["ImageBind"]
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    path_out = MODEL_FEATURES_FILES["ImageBind_AVG"]
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    pickle.dump(all_feats_avg, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    os.makedirs(FEATURES_DIR, exist_ok=True)
    extract_features()
