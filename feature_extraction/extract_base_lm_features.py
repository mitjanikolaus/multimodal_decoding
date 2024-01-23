import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pickle
from tqdm import tqdm

from feature_extraction.feat_extraction_utils import FeatureExtractor
from utils import LANG_FEAT_KEY, model_features_file_path

BATCH_SIZE = 512

if torch.cuda.is_available():
    device = "cuda"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = "cpu"


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
    # model_name = 'bert-large-uncased'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, BATCH_SIZE, device)
    # extractor.extract_features()
    #
    # model_name = 'gpt2-xl'
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model = GPT2Model.from_pretrained(model_name)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, batch_size=10, device="cpu")
    # extractor.extract_features()

    model_name = "mistralai/Mistral-7B-v0.1"    # mistralai/Mixtral-8x7B-v0.1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    extractor = LanguageModelFeatureExtractor(model, tokenizer, "mistral", batch_size=10, device="cpu")
    extractor.extract_features()

    # model_name = 'facebook/opt-30b'
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # model_kwargs = {"device_map": "auto", "load_in_8bit": True}
    # model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, "opt-30b", batch_size=10, device="cpu")
    # extractor.extract_features()
