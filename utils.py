import os
import numpy as np
import pandas as pd
from networkx.classes import all_neighbors
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from nibabel.nifti1 import intent_codes, data_type_codes

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.expanduser("~/data/multimodal_decoding")

COCO_IMAGES_DIR = os.path.expanduser("~/data/coco/")

NN_FEATURES_DIR = os.path.join(DATA_DIR, "nn_features")

STIM_INFO_PATH = os.path.join(DATA_DIR, "stimuli.p")
STIMULI_IDS_PATH = os.path.join(DATA_DIR, "stimuli_ids.p")

FMRI_DATA_DIR = os.path.join(DATA_DIR, "fmri")
FMRI_RAW_DATA_DIR = os.path.join(FMRI_DATA_DIR, "raw")
FMRI_RAW_BIDS_DATA_DIR = os.path.join(FMRI_RAW_DATA_DIR, "bids")
FMRI_PREPROCESSED_DATA_DIR = os.path.join(FMRI_DATA_DIR, "preprocessed")
FMRI_BETAS_DIR = os.path.join(FMRI_DATA_DIR, "betas")
FMRI_SURFACE_LEVEL_DIR = os.path.join(FMRI_DATA_DIR, "betas_surface_level")
FMRI_REGFILES_DIR = os.path.join(FMRI_DATA_DIR, "regfiles")

FREESURFER_BASE_DIR = os.path.join(DATA_DIR, "freesurfer")
FREESURFER_HOME_DIR = "/usr/local/freesurfer/7.4.1"

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']
HEMIS = ['left', 'right']
HEMIS_FS = ['lh', 'rh']

VISION_MEAN_FEAT_KEY = "vision_features_mean"
VISION_CLS_FEAT_KEY = "vision_features_cls"

LANG_MEAN_FEAT_KEY = "lang_features_mean"
LANG_CLS_FEAT_KEY = "lang_features_cls"

FUSED_MEAN_FEAT_KEY = "fused_mean_features"
FUSED_CLS_FEAT_KEY = "fused_cls_features"

FS_HEMI_NAMES = {'left': 'lh', 'right': 'rh'}


def nipype_subject_id(subject):
    return f'_subject_id_{subject}'


def model_features_file_path(model_name):
    return os.path.join(NN_FEATURES_DIR, f"{model_name.lower()}.p")


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

IDS_IMAGES_IMAGERY = [scene[1] for scenes_subj in IMAGERY_SCENES.values() for scene in scenes_subj]

IDS_IMAGES_TEST = [
    3862,
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
    573980
]

NUM_TEST_STIMULI = len(IDS_IMAGES_TEST) * 2
INDICES_TEST_STIM_CAPTION = list(range(NUM_TEST_STIMULI // 2))
INDICES_TEST_STIM_IMAGE = list(range(NUM_TEST_STIMULI // 2, NUM_TEST_STIMULI))
IDS_TEST_STIM = np.array(IDS_IMAGES_TEST + IDS_IMAGES_TEST)

ACC_MODALITY_AGNOSTIC = "pairwise_acc_modality_agnostic"
ACC_CAPTIONS = "pairwise_acc_captions"
ACC_IMAGES = "pairwise_acc_images"

ACC_CROSS_IMAGES_TO_CAPTIONS = "pairwise_acc_cross_images_to_captions"
ACC_CROSS_CAPTIONS_TO_IMAGES = "pairwise_acc_cross_captions_to_images"

ACC_IMAGERY = "pairwise_acc_imagery"
ACC_IMAGERY_WHOLE_TEST = "pairwise_acc_imagery_whole_test_set"


def correlation_num_voxels_acc(scores, nan_locations, n_neighbors, args):
    all_scores = []
    all_neighbors = []
    for subject in args.subjects:
        for hemi in HEMIS:
            for metric in ["captions", "images"]:
                nans = nan_locations[subject][hemi]
                all_scores.extend(scores[subject][hemi][metric][~nans])
                all_neighbors.extend(n_neighbors[subject][hemi])

    corr = pearsonr(all_neighbors, all_scores)

    df = pd.DataFrame({'n_neighbors': all_neighbors, 'scores': all_scores})
    df['n_neighbors_binned'] = pd.cut(df['n_neighbors'], bins=range(125, 1750, 250),
                                      labels=list(range(250, 1550, 250)))

    plt.figure(figsize=(20, 7))
    sns.barplot(data=df, x="n_neighbors_binned", y="scores")
    plt.xlabel("number of voxels")
    plt.ylabel("pairwise accuracy (mean)")
    plt.savefig(f"results/searchlight_num_voxels_correlations/searchlight_correlation_num_voxels_acc.png",
                dpi=300)
    print(df.groupby('n_neighbors_binned').aggregate({"scores": "mean"}))

    sns.histplot(x=all_neighbors, y=all_scores)
    plt.xlabel("number of voxels")
    plt.ylabel("pairwise accuracy (mean)")
    plt.title(f"pearson r: {corr[0]:.2f} | p = {corr[1]}")
    plt.savefig(f"results/searchlight_num_voxels_correlations/searchlight_correlation_num_voxels_acc_hist.png",
                dpi=300)


def export_to_gifti(scores, path):
    data = scores.astype(np.float32)
    gimage = GiftiImage(
        darrays=[GiftiDataArray(
            data,
            intent=intent_codes.code['NIFTI_INTENT_NONE'],
            datatype=data_type_codes.code['NIFTI_TYPE_FLOAT32'])]
    )
    gimage.to_filename(path)
