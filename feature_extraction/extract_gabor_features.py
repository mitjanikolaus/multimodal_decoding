import math
import os
import pickle

import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm

import moten
from PIL import Image

from feature_extraction.feat_extraction_utils import FeatureExtractor, CoCoDataset
from utils import COCO_IMAGES_DIR, STIM_INFO_PATH, STIMULI_IDS_PATH, model_features_file_path

BATCH_SIZE = 1000
N_JOBS = 20
IMG_SIZE = 256

class GaborFeatureExtractor(FeatureExtractor):
    def __init__(self, model_name=None, batch_size=BATCH_SIZE):
        super(FeatureExtractor).__init__()
        print(f"Feature extraction for {model_name}")

        self.model_name = model_name

        self.ds = CoCoDataset(COCO_IMAGES_DIR, STIM_INFO_PATH, STIMULI_IDS_PATH, 'image')
        self.dloader = DataLoader(self.ds, shuffle=False, batch_size=batch_size)

    def extract_features(self):
        all_feats = dict()
        for ids, captions, img_paths in tqdm(self.dloader):
            ids = [id.item() for id in ids]

            def extract_feats(img_paths):
                feats = []
                for img_path in img_paths:
                    image = Image.open(img_path).convert('RGB')

                    # resize so that the width is 256 pixels
                    resized = image.resize((IMG_SIZE, round((image.height / image.width) * IMG_SIZE)))

                    # center-crop 256 pixels of height
                    cropped = resized.crop(
                        (0, round((resized.height - IMG_SIZE) / 2), resized.width, round((resized.height + IMG_SIZE) / 2)))

                    luminance_image = moten.io.imagearray2luminance(np.asarray(cropped))

                    pyramid = moten.get_default_pyramid(vhsize=(luminance_image.shape[1], luminance_image.shape[2]),
                                                        temporal_frequencies=[0])

                    moten_features = pyramid.project_stimulus(luminance_image, spatial_only=True)
                    feats.append(moten_features)

                return np.array(feats)

            n_per_job = math.ceil(len(img_paths) / N_JOBS)
            batch_feats = Parallel(n_jobs=N_JOBS)(
                delayed(extract_feats)(
                    img_paths[i * n_per_job:(i + 1) * n_per_job],
                )
                for i in range(N_JOBS)
            )
            batch_feats = np.concatenate(batch_feats)

            for id, feats in zip(ids, batch_feats):
                all_feats[id] = feats

        path_out = model_features_file_path(self.model_name)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    extractor = GaborFeatureExtractor("gabor", BATCH_SIZE)
    extractor.extract_features()
