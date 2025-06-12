import math
import os
import pickle

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from skimage.filters import gabor_kernel
from torch.utils.data import DataLoader
from tqdm import tqdm

import moten
from PIL import Image, ImageDraw, ImageFont

from data import LANG_CLS_FEAT_KEY, VISION_CLS_FEAT_KEY
from feature_extraction.feat_extraction_utils import FeatureExtractor, CoCoDataset
from utils import COCO_IMAGES_DIR, STIM_INFO_PATH, model_features_file_path

BATCH_SIZE = 1000
N_JOBS = 20
IMG_SIZE = 256

FONT_NAME = "YaHei.Consolas.1.12.ttf"
FONT_SIZE = 6
TEXT_COLOR = "white"
TEXT_BACKGROUND = "grey"
MAX_CAPTION_LEN = 70


def make_gabor_filterbank():
    kernels = []
    for theta in range(8):
        theta = theta / 8.0 * np.pi
        sigma_values = (1, 3, 5, 7)
        for sigma in sigma_values:
            for frequency in [0, 2, 4, 8, 16]:
                kernel = np.real(
                    gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                )
                kernels.append(kernel)
    return kernels


def compute_gabor_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndimage.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats.flatten()


class GaborFeatureExtractor(FeatureExtractor):
    def __init__(self, model_name=None, batch_size=BATCH_SIZE):
        super(FeatureExtractor).__init__()
        print(f"Feature extraction for {model_name}")

        self.model_name = model_name

        self.ds = CoCoDataset(COCO_IMAGES_DIR, STIM_INFO_PATH, 'both')
        self.dloader = DataLoader(self.ds, shuffle=False, batch_size=batch_size)

    def extract_features(self):
        all_feats = dict()
        for ids, captions, img_paths in tqdm(self.dloader):
            ids = [id.item() for id in ids]

            def extract_feats(paths, caps):
                kernels = make_gabor_filterbank()

                feats_imgs = []
                feats_caps = []
                for img_path, caption in zip(paths, caps):
                    image = Image.open(img_path).convert('RGB')

                    # resize so that the width is 256 pixels
                    resized = image.resize((IMG_SIZE, round((image.height / image.width) * IMG_SIZE)))

                    # center-crop 256 pixels of height
                    cropped = resized.crop(
                        (0, round((resized.height - IMG_SIZE) / 2),
                         resized.width, round((resized.height + IMG_SIZE) / 2))
                    )

                    grayscale_image = cropped.convert('L')
                    feats_img = compute_gabor_feats(grayscale_image, kernels)

                    # luminance_image = moten.io.imagearray2luminance(np.asarray(cropped))

                    # pyramid = moten.get_default_pyramid(vhsize=(luminance_image.shape[1], luminance_image.shape[2]),
                    #                                     temporal_frequencies=[0])
                    #
                    # moten_features_img = pyramid.project_stimulus(luminance_image, spatial_only=True)
                    #
                    feats_imgs.append(feats_img.squeeze())

                    # make image from caption
                    caption = caption[:MAX_CAPTION_LEN]
                    font = ImageFont.truetype(FONT_NAME, FONT_SIZE)
                    text_width = int(getSize(caption, font))

                    if not text_width <= IMG_SIZE:
                        print(f"Warning: caption longer than image width! {text_width}")
                        print(caption)

                    image_caption = Image.new('RGB', (IMG_SIZE, IMG_SIZE), TEXT_BACKGROUND)
                    d = ImageDraw.Draw(image_caption)
                    d.text(((IMG_SIZE - text_width) / 2, IMG_SIZE / 2), caption, fill=TEXT_COLOR, font=font)

                    # luminance_image_caption = moten.io.imagearray2luminance(np.asarray(image_caption))
                    #
                    # pyramid = moten.get_default_pyramid(
                    #     vhsize=(luminance_image_caption.shape[1], luminance_image_caption.shape[2]),
                    #     temporal_frequencies=[0])
                    #
                    # moten_features_caption = pyramid.project_stimulus(luminance_image_caption, spatial_only=True)

                    grayscale_image = image_caption.convert('L')
                    feats_cap = compute_gabor_feats(grayscale_image, kernels)

                    feats_caps.append(feats_cap.squeeze())

                return np.array(feats_imgs), np.array(feats_caps)

            n_per_job = math.ceil(len(img_paths) / N_JOBS)
            batch_feats = Parallel(n_jobs=N_JOBS)(
                delayed(extract_feats)(
                    img_paths[i * n_per_job:(i + 1) * n_per_job],
                    captions[i * n_per_job:(i + 1) * n_per_job],
                )
                for i in range(N_JOBS)
            )
            batch_feats = np.concatenate(batch_feats)
            batch_feats_imgs = np.concatenate([batch_feat[0] for batch_feat in batch_feats])
            batch_feats_caps = np.concatenate([batch_feat[1] for batch_feat in batch_feats])

            for id, feats_img, feats_cap in zip(ids, batch_feats_imgs, batch_feats_caps):
                all_feats[id] = dict()
                all_feats[id][VISION_CLS_FEAT_KEY] = feats_img
                all_feats[id][LANG_CLS_FEAT_KEY] = feats_cap

        path_out = model_features_file_path(self.model_name)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def getSize(txt, font):
    testImg = Image.new('RGB', (1, 1))
    testDraw = ImageDraw.Draw(testImg)
    return testDraw.textlength(txt, font)


if __name__ == "__main__":
    # img_path = '/home/mitja/Downloads/cross_images_to_captions.png'
    # image = Image.open(img_path).convert('RGB')
    #
    # # resize so that the width is 256 pixels
    # resized = image.resize((IMG_SIZE, round((image.height / image.width) * IMG_SIZE)))
    #
    # # center-crop 256 pixels of height
    # cropped = resized.crop(
    #     (0, round((resized.height - IMG_SIZE) / 2),
    #      resized.width, round((resized.height + IMG_SIZE) / 2))
    # )
    #
    # grayscale_image = cropped.convert('L')
    # kernels = make_gabor_filterbank()
    # feats = compute_gabor_feats(grayscale_image, kernels)
    # print(feats.shape)
    # print(feats)
    # luminance_image = moten.io.imagearray2luminance(np.asarray(cropped))
    #
    #
    # pyramid = moten.get_default_pyramid(vhsize=(luminance_image.shape[1], luminance_image.shape[2]),
    #                                     temporal_frequencies=[0])
    #
    # moten_features_img = pyramid.project_stimulus(luminance_image, spatial_only=True)

    # caption = "The man on the skateboard and the dog are getting their picture taken"
    # font = ImageFont.truetype(FONT_NAME, FONT_SIZE)
    # text_width = int(getSize(caption, font))
    #
    # if not text_width <= IMG_SIZE:
    #     print(f"Warning: caption longer than image width! {text_width}")
    #     print(caption)
    #
    # image_caption = Image.new('RGB', (IMG_SIZE, IMG_SIZE), TEXT_BACKGROUND)
    # d = ImageDraw.Draw(image_caption)
    # d.text(((IMG_SIZE - text_width) / 2, IMG_SIZE/2), caption, fill=TEXT_COLOR, font=font)
    #
    # luminance_image_caption = moten.io.imagearray2luminance(np.asarray(image_caption))
    #
    # pyramid = moten.get_default_pyramid(
    #     vhsize=(luminance_image_caption.shape[1], luminance_image_caption.shape[2]),
    #     temporal_frequencies=[0])
    #
    # moten_features_caption = pyramid.project_stimulus(luminance_image_caption, spatial_only=True)
    #
    #
    # plt.imshow(image_caption)
    # plt.show()

    extractor = GaborFeatureExtractor("gabor_simple", BATCH_SIZE)
    extractor.extract_features()
