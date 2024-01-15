import os
import pickle
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

DATA_DIR = os.path.expanduser("~/data/multimodal_decoding")
FEATURES_DIR = os.path.join(DATA_DIR, "feats_unimodal")

COCO_2017_TRAIN_IMAGES_DIR = os.path.expanduser("~/data/coco/images/train_2017")

STIMULI_IDS_PATH = os.path.join(DATA_DIR, "stimuli_ids.p")
CAPTIONS_PATH = os.path.join(DATA_DIR, "ShortCaptions_AllStims_CorrectedTyposInTestCaptionsLinux.p")
FMRI_DATA_DIR = os.path.expanduser("~/data/multimodal_decoding/fmri/")

TWO_STAGE_GLM_DATA_DIR = os.path.join(FMRI_DATA_DIR, "glm_manual/two-stage-mni/")

SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']

PCA_NUM_COMPONENTS = 768

MODEL_FEATURES_FILES = {
    "RESNET152_AVGPOOL_PCA768": os.path.join(FEATURES_DIR, "resnet/resnet152_avgpool_selected_coco_crop_pca_768.p"),
    "RESNET152_AVGPOOL": os.path.join(FEATURES_DIR, "resnet/resnet152_avgpool_selected_coco_crop.p"),
    "GPT2XL_AVG_PCA768": os.path.join(FEATURES_DIR, "gpt/gpt2_xl_avg_selected_coco_pca_768.p"),
    "GPT2XL_AVG": os.path.join(FEATURES_DIR, "gpt/gpt2_xl_avg_selected_coco.p"),
    "VITL16_ENCODER": os.path.join(FEATURES_DIR, "vit/vit_l_16_encoder_selected_coco_crop.p"), 
    "VITL16_ENCODER_PCA768": os.path.join(FEATURES_DIR, "vit/vit_l_16_encoder_selected_coco_crop_pca_768.p"),
    "CLIP_L": os.path.join(FEATURES_DIR, "clip/clip_l_VITL14336px_selected_coco_dataset_crop.p"),
    "CLIP_V": os.path.join(FEATURES_DIR, "clip/clip_v_VITL14336px_selected_coco_dataset_crop.p"),
    "CLIP_L_PCA768": os.path.join(FEATURES_DIR, "clip/clip_l_VITL14336px_selected_coco_dataset_crop_pca_768.p"),
    "CLIP_V_PCA768": os.path.join(FEATURES_DIR, "clip/clip_v_VITL14336px_selected_coco_dataset_crop_pca_768.p"),
    "BERT_LARGE": os.path.join(FEATURES_DIR, "bert/bert_large_avg_selected_coco.p"),
    "VisualBERT": os.path.join(FEATURES_DIR, "visualbert/visualbert_vl_avg.p"),
    "VILT": os.path.join(FEATURES_DIR, "vilt/vilt_vl_concatenated.p"),
    "VILT_AVG": os.path.join(FEATURES_DIR, "vilt/vilt_vl_averaged.p"),
    "ImageBind": os.path.join(FEATURES_DIR, "imagebind/imagebind_vl_concatenated.p"),
    "ImageBind_AVG": os.path.join(FEATURES_DIR, "imagebind/imagebind_vl_avg.p")
}


IMAGES_IMAGERY_CONDITION = [
    [406591, f'images/train2017/000000406591.jpg',
     'A woman sits in a beach chair as a man walks along the sand'],
    [324670, f'images/train2017/000000324670.jpg',
     'White bird sitting in front of a lighthouse with a red roof'],
    [563723, f'images/train2017/000000563723.jpg',
     'A little cat standing on the lap of a man sitting in a chair'],
    [254568, f'images/train2017/000000254568.jpg',
     'A lonely giraffe is walking in the middle of a grassy field'],
    [42685, f'images/train2017/000000042685.jpg',
     'A party of young people in a bedroom with a large box of pizza'],
    [473773, f'images/train2017/000000473773.jpg',
     'A man walking across a field of snow holding skis and ski poles'],
    [485909, f'images/train2017/000000485909.jpg',
     'Two men are discussing something next to a desk with a laptop'],
    [544502, f'images/train2017/000000544502.jpg',
     'A young male holding a racket and tennis ball in a tennis court'],
    [556512, f'images/train2017/000000556512.jpg',
     'A group of kids swimming in the ocean around a man on a surfboard'],
    [410573, f'images/train2017/000000410573.jpg',
     'A cat and a dog staring intensely at each other on an armchair'],
    [48670, f'images/train2017/000000048670.jpg',
     'A man stands by a rainy street with an umbrella over his head'],
    [263212, f'images/train2017/000000263212.jpg',
     'A woman working on her computer while also using her cell phone'],
    [214816, f'images/train2017/000000214816.jpg',
     'An old broken down church and graffiti on surrounding structures'],
    [141343, f'images/train2017/000000141343.jpg',
     'A teddy bear lying down on the sidewalk in front of a store'],
    [213506, f'images/train2017/000000213506.jpg',
     'A woman leaning out a window to talk to someone on the sidewal'],
    [162396, f'images/train2017/000000162396.jpg',
     'The man on the skateboard and the dog are getting their picture taken'],
]


IMAGERY_SCENES = {
    'sub-01':
        [
            ('A woman sits in a beach chair as a man walks along the sand', 406591),
            ('White bird sitting in front of a lighthouse with a red roof', 324670),
            ('A little cat standing on the lap of a man sitting in a chair', 563723),
        ],

    'sub-02':
        [
            ("A woman sits in a beach chair as a man walks along the sand", 406591),
            ("A little cat standing on the lap of a man sitting in a chair", 563723),
            ("A lonely giraffe is walking in the middle of a grassy field", 254568)
        ],

    'sub-03':
        [
            ("A party of young people in a bedroom with a large box of pizza", 42685),
            ("A man walking across a field of snow holding skis and ski poles", 473773),
            ("Two men are discussing something next to a desk with a laptop", 485909),
        ],

    'sub-04':
        [
            ('A young male holding a racket and tennis ball in a tennis court', 544502),
            ('A group of kids swimming in the ocean around a man on a surfboard', 556512),
            ('A cat and a dog staring intensely at each other on an armchair', 410573),
        ],

    'sub-05':
        [
            ('A man stands by a rainy street with an umbrella over his head', 48670),
            ('A woman working on her computer while also using her cell phone', 263212),
            ('An old broken down church and graffiti on surrounding structures', 214816),
        ],

    'sub-07':
        [
            ('A teddy bear lying down on the sidewalk in front of a store', 141343),
            ('A woman leaning out a window to talk to someone on the sidewal', 213506),
            ('The man on the skateboard and the dog are getting their picture taken', 162396),
        ],
}


IMAGES_TEST = [3862,
               6450,
               16764,
               25902,
               38938,
               43966,
               47566,
               53580,
               55413,
               57703,
               63881,
               70426,
               79642,
               122403,
               133449,
               138529,
               146411,
               159225,
               163240,
               165419,
               165874,
               176509,
               180389,
               183210,
               186308,
               186788,
               192095,
               195406,
               201601,
               211189,
               220654,
               221313,
               238193,
               252018,
               255714,
               271844,
               275995,
               278135,
               279331,
               287434,
               292888,
               310552,
               315083,
               318108,
               323797,
               353260,
               363483,
               367120,
               380011,
               385795,
               388398,
               414373,
               423618,
               450719,
               454636,
               457249,
               466514,
               467854,
               475693,
               499733,
               505655,
               512289,
               534975,
               536798,
               546074,
               548167,
               555739,
               560282,
               567627,
               573980]


class COCOSelected(Dataset):
    r"""
    Pytorch dataset that loads the preselected data from the COCO dataset.
    The preselected data are given in a separate file (`selection_file`).
    """

    def __init__(self, coco_root, captions_path, stimuli_ids_path, mode='image', transform=None):
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
        self.transform = transform

    def __len__(self):
        return len(self.stimuli_ids)

    def __getitem__(self, index):
        id = self.stimuli_ids[index]
        if self.mode == 'image':
            img_path = os.path.join(self.root, self.img_paths[id])
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, id, img_path

        elif self.mode == 'caption':
            cap = self.captions[id]
            if self.transform is not None:
                cap = self.transform(cap)
            return cap, id

        elif self.mode == 'both':
            img_path = os.path.join(self.root, self.img_paths[id])
            cap = self.captions[id]
            if self.transform is not None:
                cap = self.transform(cap)
            return id, cap, img_path
