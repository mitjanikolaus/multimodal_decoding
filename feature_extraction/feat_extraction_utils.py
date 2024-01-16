import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import IMAGES_IMAGERY_CONDITION, COCO_2017_TRAIN_IMAGES_DIR, CAPTIONS_PATH, STIMULI_IDS_PATH, FEATURES_DIR, \
    LING_FEAT_KEY, VISION_FEAT_KEY, model_features_file_path


class COCOSelected(Dataset):
    r"""
    Pytorch dataset that loads the preselected data from the COCO dataset.
    The preselected data are given in a separate file (`selection_file`).
    """

    def __init__(self, coco_root, captions_path, stimuli_ids_path, mode='image'):
        r"""
        Args:
            `coco_root` (str): address to the coco2017 root folder (= the parent directory of `images` folder)
            `selection_file` (pickle): address to file containing information about the preselected coco entries
            `mode` (str): can be `caption` or `image` to load captions or images, respectively. Default: `image`
            `transform` (callable): data transformation. Default: None
        """
        super().__init__()
        data = np.load(captions_path, allow_pickle=True)
        data.extend(IMAGES_IMAGERY_CONDITION)
        self.stimuli_ids = pickle.load(open(stimuli_ids_path, "rb"))
        self.img_paths = {id: path for id, path, _ in data if id in self.stimuli_ids}
        self.captions = {id: caption for id, _, caption in data if id in self.stimuli_ids}
        self.root = coco_root
        self.mode = mode

    def __len__(self):
        return len(self.stimuli_ids)

    def __getitem__(self, index):
        id = self.stimuli_ids[index]
        if self.mode == 'image':
            img_path = os.path.join(self.root, self.img_paths[id])
            img = Image.open(img_path).convert('RGB')
            return img, id, img_path

        elif self.mode == 'caption':
            cap = self.captions[id]
            return cap, id

        elif self.mode == 'both':
            img_path = os.path.join(self.root, self.img_paths[id])
            cap = self.captions[id]
            return id, cap, img_path


class FeatureExtractor:
    def __init__(self, model, prepocessor=None, model_name=None, batch_size=10, device="cpu"):
        super().__init__()
        print(f"Feature extraction on {device}")

        self.device = device

        self.model = model.to(device)
        self.model.eval()

        self.preprocessor = prepocessor

        self.model_name = model_name

        self.ds = COCOSelected(COCO_2017_TRAIN_IMAGES_DIR, CAPTIONS_PATH, STIMULI_IDS_PATH, 'both')
        self.dloader = DataLoader(self.ds, shuffle=False, batch_size=batch_size)

        os.makedirs(FEATURES_DIR, exist_ok=True)

    def extract_features(self):
        all_feats = dict()
        for ids, captions, img_paths in tqdm(self.dloader):
            ids = [id.item() for id in ids]
            language_feats_batch, vision_feats_batch = self.extract_features_from_batch(ids, captions, img_paths)
            for id, feats_lang, feats_vision, path in zip(ids, language_feats_batch, vision_feats_batch, img_paths):
                if feats_lang.device != "cpu":
                    feats_lang = feats_lang.cpu()
                if feats_vision.device != "cpu":
                    feats_vision = feats_vision.cpu()

                all_feats[id] = {LING_FEAT_KEY: feats_lang.numpy(), VISION_FEAT_KEY: feats_vision.numpy()}

        path_out = model_features_file_path(self.model_name)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def extract_features_from_batch(self, ids, captions, img_paths):
        raise NotImplementedError()
