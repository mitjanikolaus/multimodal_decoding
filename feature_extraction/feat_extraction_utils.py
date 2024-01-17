import os
import pickle

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import IMAGES_IMAGERY_CONDITION, COCO_2017_TRAIN_IMAGES_DIR, CAPTIONS_PATH, STIMULI_IDS_PATH, FEATURES_DIR, \
    LANG_FEAT_KEY, VISION_FEAT_KEY, model_features_file_path, IMAGES_TEST


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
        print(f"Feature extraction for {model_name} on {device}")

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
            if language_feats_batch is None:
                language_feats_batch = [None] * len(ids)
            else:
                language_feats_batch = language_feats_batch.cpu().numpy()
            if vision_feats_batch is None:
                vision_feats_batch = [None] * len(ids)
            else:
                vision_feats_batch = vision_feats_batch.cpu().numpy()
            for id, feats_lang, feats_vision, path in zip(ids, language_feats_batch, vision_feats_batch, img_paths):
                all_feats[id] = {LANG_FEAT_KEY: feats_lang, VISION_FEAT_KEY: feats_vision}

        path_out = model_features_file_path(self.model_name)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def extract_features_from_batch(self, ids, captions, img_paths):
        raise NotImplementedError()


def apply_pca(n_components, data_pickle_file):
    r"""
    Applies PCA on the given latent vectors. The function saves the following (pickle) files
    in the same directory as the given pickle file for the latent vectors:

    - PCA module (to be used later to recover original vectors)
    - Scaler module (that used for data normalization before the PCA)
    - Compressed vectors (transformed latent vectors)

    Args:
        `n_components` (int): number of PCs
        `data_pickle_file` (str): address to the pickle file containing latent vectors
    """
    print("Performing PCA..")
    excluded = set(IMAGES_TEST + [a[0] for a in IMAGES_IMAGERY_CONDITION])

    filename = os.path.basename(data_pickle_file)
    dirname = os.path.dirname(data_pickle_file)
    base_filename = filename[:filename.find('.p')]

    with open(data_pickle_file, 'rb') as f:
        data = pickle.load(f)

    # gathering all the stim ids
    stim_ids = list(data.keys())

    # finding keys for feature values
    features_names = [k for k in data[stim_ids[0]].keys() if "feature" in k]

    # normalizing first, then PCA
    transformed_data = {k: {} for k in stim_ids}
    for feature in features_names:
        # creating the combined vector with order of stim_ids
        vectors = []
        all_vectors = []
        for sid in stim_ids:
            if sid not in excluded:  # no influence of the test and imagery sets
                vectors.append(data[sid][feature])

            all_vectors.append(data[sid][feature])

        vectors = np.array(vectors)
        all_vectors = np.array(all_vectors)

        scaler = StandardScaler()
        normalized = scaler.fit_transform(vectors)
        normalized_all_vectors = scaler.transform(all_vectors)

        pca = PCA(n_components=n_components, whiten=True)  # whiten=True: divide by singular values
        pca.fit(normalized)
        reduced_all_vectors = pca.transform(normalized_all_vectors)

        # storing the reduced vectors (keeping stim_id ordering)
        for idx, sid in enumerate(stim_ids):
            transformed_data[sid][feature] = reduced_all_vectors[idx]

        # save PCA and scaler modules for future transformations
        pca_file = os.path.join(dirname, f"{base_filename}_pca_{n_components}_pca_{feature}_module.p")
        with open(pca_file, 'wb') as handle:
            pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
        scaler_file = os.path.join(dirname, f"{base_filename}_pca_{n_components}_scaler_{feature}_module.p")
        with open(scaler_file, 'wb') as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # filling the rest as before
    for sid in stim_ids:
        for k in data[sid]:
            if k not in transformed_data[sid]:
                transformed_data[sid][k] = data[sid][k]

    # saving transformed data in the same place with PCA_ncomp postfix
    new_file = os.path.join(dirname, f"{base_filename}_pca_{n_components}.p")

    with open(new_file, 'wb') as handle:
        pickle.dump(transformed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

