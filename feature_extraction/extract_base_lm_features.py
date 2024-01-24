import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, \
    MistralModel, LlamaTokenizer, LlamaModel

from feature_extraction.feat_extraction_utils import FeatureExtractor
from utils import DATA_DIR

BATCH_SIZE = 512

if torch.cuda.is_available():
    device = "cuda"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = "cpu"


class LanguageModelFeatureExtractor(FeatureExtractor):

    def extract_features_from_batch(self, ids, captions, img_paths):
        if self.preprocessor.pad_token is None:
            self.preprocessor.pad_token = tokenizer.eos_token
            self.preprocessor.pad_token_id = tokenizer.eos_token_id

        inputs = self.preprocessor(text=captions, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        # Average while ignoring padding tokens
        mask = inputs.data["attention_mask"]
        mask_expanded = mask.unsqueeze(-1).expand((mask.shape[0], mask.shape[1], last_hidden_state.shape[-1]))
        last_hidden_state[mask_expanded == 0] = 0
        feats_lang = last_hidden_state.sum(axis=1) / mask_expanded.sum(dim=1)

        return feats_lang, None, None


if __name__ == "__main__":
    # model_name = 'bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, BATCH_SIZE, device)
    # extractor.extract_features()

    # model_name = 'bert-large-uncased'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, BATCH_SIZE, device)
    # extractor.extract_features()

    # model_name = 'gpt2-xl'
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # model = GPT2Model.from_pretrained(model_name)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, batch_size=10, device="cpu")
    # extractor.extract_features()

    model_name = 'gpt2-large'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, batch_size=BATCH_SIZE, device=device)
    extractor.extract_features()

    model_name = 'gpt2-medium'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    extractor = LanguageModelFeatureExtractor(model, tokenizer, model_name, batch_size=BATCH_SIZE, device=device)
    extractor.extract_features()

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    extractor = LanguageModelFeatureExtractor(model, tokenizer, "gpt2-small", batch_size=BATCH_SIZE, device=device)
    extractor.extract_features()

    # model_name = "mistralai/Mistral-7B-v0.1"    # mistralai/Mixtral-8x7B-v0.1
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = MistralModel.from_pretrained(model_name)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, "mistral", batch_size=10, device="cpu")
    # extractor.extract_features()

    # model_weights_path = os.path.join(DATA_DIR, "llama2_7b")
    # tokenizer = LlamaTokenizer.from_pretrained(model_weights_path)
    # model = LlamaModel.from_pretrained(model_weights_path)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, "llama2-7b", batch_size=10, device="cpu")
    # extractor.extract_features()

    # model_name = 'facebook/opt-30b'
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # model_kwargs = {"device_map": "auto", "load_in_8bit": True}
    # model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    # extractor = LanguageModelFeatureExtractor(model, tokenizer, "opt-30b", batch_size=10, device="cpu")
    # extractor.extract_features()
