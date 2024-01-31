############################################
# Training multimodal linear brain decoders
# inputs can be of any modality
# outputs are uni-modal
############################################
import argparse
import re
import time

import numpy as np
import nibabel as nib
from nilearn.datasets import fetch_atlas_destrieux_2009
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import os
from glob import glob
import pickle
from decoding_utils import get_distance_matrix
from tqdm import trange

from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, model_features_file_path, VISION_MEAN_FEAT_KEY, \
    VISION_CLS_FEAT_KEY, LANG_FEAT_KEY, \
    MULTIMODAL_FEAT_KEY
from analyses.ridge_regression_decoding import REGIONS_OCCIPITAL_EXCLUSIVE, REGIONS_HIGH_LEVEL_VISUAL, REGIONS_LANGUAGE
import pandas as pd

def run():
    long_names = {}
    ids = {}
    with open("destrieux.txt") as file:
        for line in file:
            line = line.rstrip()
            long_name = " ".join(line.split(', ')[1].split(' ')[1:-1])
            pattern = r'[0-9]'
            long_name = re.sub(pattern, '', long_name)
            name = line.split(', ')[1].split(' ')[0]
            long_names[name] = long_name
            ids[name] = int(line.split(',')[0])

    destrieux_atlas = fetch_atlas_destrieux_2009()
    label_to_value_dict = {label[1]: int(label[0]) for label in destrieux_atlas['labels']}
    atlas_map = nib.load(destrieux_atlas.maps).get_fdata()

    for name, region in zip(["low_level_visual", "high_level_visual", "language_network"], [REGIONS_OCCIPITAL_EXCLUSIVE, REGIONS_HIGH_LEVEL_VISUAL, REGIONS_LANGUAGE]):
        region_names = [label for label in region]
        values = [label_to_value_dict[label] for label in region_names]
        roi_mask = np.isin(atlas_map, values)
        roi_mask_size = roi_mask.sum()
        print(name)
        print("ROI size: {} voxels".format(roi_mask_size))
        print("Parts:")
        for name, id in zip(region_names, values):
            region_mask = np.isin(atlas_map, id)
            region_mask_size = region_mask.sum()
            print(f"\tID: {ids[name[2:]]:02d} num voxels: {region_mask_size:04d} names: {name}:\t({name[0]} {long_names[name[2:]]})")
        print()


if __name__ == "__main__":
    run()
