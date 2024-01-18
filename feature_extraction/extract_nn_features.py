import os
import torch
from transformers import pipeline, ViTModel, ViTImageProcessor, AutoFeatureExtractor, ResNetModel, BertModel, \
    BertTokenizer, GPT2Tokenizer, GPT2Model
import numpy as np
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm

from feature_extraction.feat_extraction_utils import FeatureExtractor
from utils import SUBJECTS, IMAGERY_SCENES, STIMULI_IDS_PATH, TWO_STAGE_GLM_DATA_DIR, LANG_FEAT_KEY, \
    model_features_file_path

BATCH_SIZE = 256

if torch.cuda.is_available():
    device = "cuda"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = "cpu"


def load_and_save_relevant_coco_ids():
    if not os.path.isfile(STIMULI_IDS_PATH):
        print("Extracting stimulus ids (coco ids)")
        all_ids = []
        for mode in ["train", "test", "imagery"]:
            for subject in SUBJECTS:
                fmri_root_dir = os.path.join(TWO_STAGE_GLM_DATA_DIR, subject)
                imagery_scenes = IMAGERY_SCENES[subject]

                fmri_betas_addresses = sorted(glob(os.path.join(fmri_root_dir, f'betas_{mode}*', '*.nii')))

                for addr in tqdm(fmri_betas_addresses):
                    file_name = os.path.basename(addr)
                    if 'I' in file_name:  # Image
                        stim_id = int(file_name[file_name.find('I') + 1:-4])
                    elif 'C' in file_name:  # Caption
                        stim_id = int(file_name[file_name.find('C') + 1:-4])
                    else:  # imagery
                        id = int(file_name[file_name.find('.nii') - 1:-4])
                        stim_id = imagery_scenes[id - 1][1]
                    all_ids.append(stim_id)

        all_ids = sorted(list(set(all_ids)))
        pickle.dump(all_ids, open(STIMULI_IDS_PATH, "wb"))


class ViTFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = self.preprocessor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        feats_vision = last_hidden_state[:, 0, :]
        return None, feats_vision


class ResNetFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        images = [Image.open(path) for path in img_paths]
        images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]

        inputs = self.preprocessor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        feats_vision = outputs.pooler_output.squeeze()
        return None, feats_vision


class LanguageModelFeatureExtractor(FeatureExtractor):

    def extract_features(self):
        all_feats = dict()
        model = pipeline('feature-extraction', model=self.model_name, device=self.device)
        with tqdm(self.dloader, unit="batch") as tepoch:
            for ids, captions, _ in tepoch:
                ids = [id.item() for id in ids]

                with torch.no_grad():
                    feats_batch = model(list(captions))
                for feats, id, caption in zip(feats_batch, ids, captions):
                    feats_mean = np.array(feats[0]).mean(axis=0)
                    all_feats[id] = {LANG_FEAT_KEY: feats_mean.squeeze()}

        path_out = model_features_file_path(self.model_name)
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        pickle.dump(all_feats, open(path_out, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # load_and_save_relevant_coco_ids()

    # model_name = 'microsoft/resnet-152'
    # feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    # model = ResNetModel(ResNetModel.from_pretrained(model_name).config)
    # extractor = ResNetFeatureExtractor(model, feature_extractor, "Resnet-152-random", BATCH_SIZE, device)
    # extractor.extract_features()

    # model_name = 'google/vit-large-patch16-224'
    # feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    # model = ViTModel.from_pretrained(model_name)
    # extractor = ViTFeatureExtractor(model, feature_extractor, "ViT_L_16", BATCH_SIZE, device)
    # extractor.extract_features()
    #
    # model_name = 'microsoft/resnet-152'
    # feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    # model = ResNetModel.from_pretrained(model_name)
    # extractor = ResNetFeatureExtractor(model, feature_extractor, "Resnet-152", BATCH_SIZE, device)
    # extractor.extract_features()

    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, BATCH_SIZE, device)
    extractor.extract_features()

    # model_name = 'gpt2-xl'
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model = GPT2Model.from_pretrained(model_name)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, batch_size=10, device="cpu")
    # extractor.extract_features()
