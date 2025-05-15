import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from nibabel import GiftiImage
from nibabel.gifti import GiftiDataArray
from nibabel.nifti1 import intent_codes, data_type_codes

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.expanduser("~/data/multimodal_decoding")

COCO_IMAGES_DIR = os.path.expanduser("~/data/coco/")

LATENT_FEATURES_DIR = os.path.join(DATA_DIR, "nn_features")
LATENT_FEATURES_NORMALIZATIONS_DIR = os.path.join(LATENT_FEATURES_DIR, "normalizations")


STIM_INFO_PATH = os.path.join(DATA_DIR, "stimuli.p")
STIMULI_IDS_PATH = os.path.join(DATA_DIR, "stimuli_ids.p")

RESULTS_FILE = "results.csv"
PREDICTIONS_FILE = "predictions.p"

FMRI_DATA_DIR = os.path.join(DATA_DIR, "fmri")
FMRI_RAW_DATA_DIR = os.path.join(FMRI_DATA_DIR, "raw")
FMRI_ANAT_DATA_DIR = os.path.join(FMRI_RAW_DATA_DIR, 'corrected_anat')
FMRI_RAW_BIDS_DATA_DIR = os.path.join(FMRI_RAW_DATA_DIR, "bids")
FMRI_PREPROCESSED_DATA_DIR = os.path.join(FMRI_DATA_DIR, "preprocessed")
FMRI_PREPROCESSING_DATASINK_DIR = os.path.join(FMRI_PREPROCESSED_DATA_DIR, "datasink")
FMRI_BETAS_DIR = os.path.join(FMRI_DATA_DIR, "betas")
FMRI_NORMALIZATIONS_DIR = os.path.join(FMRI_DATA_DIR, "normalizations")
FMRI_BETAS_SURFACE_DIR = os.path.join(FMRI_BETAS_DIR, "surface")
FMRI_STIM_INFO_DIR = os.path.join(FMRI_DATA_DIR, "stim_info")

FREESURFER_BASE_DIR = os.path.join(DATA_DIR, "freesurfer")
FREESURFER_SUBJECTS_DIR = os.path.join(FREESURFER_BASE_DIR, "subjects_downsampled_2mm")

FREESURFER_HOME_DIR = "/usr/local/freesurfer/7.4.1"

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
RIDGE_DECODER_OUT_DIR = os.path.join(DATA_DIR, "whole_brain_decoding")

SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']
HEMIS = ['left', 'right']
HEMIS_FS = ['lh', 'rh']

FS_HEMI_NAMES = {'left': 'lh', 'right': 'rh'}

DEFAULT_MODEL = "imagebind"
DEFAULT_RESOLUTION = "fsaverage7"

METRIC_CAPTIONS_DIFF_MOD_AGNO_MOD_SPECIFIC = 'diff_captions_agno_captions_specific'
METRIC_IMAGES_DIFF_MOD_AGNO_MOD_SPECIFIC = 'diff_imgs_agno_imgs_specific'
METRIC_DIFF_MOD_AGNOSTIC_MOD_SPECIFIC = 'diff_mod_agno_mod_specific'
METRIC_CROSS_DECODING = 'cross_decoding'
METRIC_MOD_AGNOSTIC_AND_CROSS = 'mod_agnostic_and_cross'

ATTENTION_MOD_DATA_DIR = os.path.join(DATA_DIR, "attention_modulation")
ATTENTION_MOD_FMRI_DATA_DIR = os.path.join(ATTENTION_MOD_DATA_DIR, "fmri")
ATTENTION_MOD_FMRI_RAW_BIDS_DATA_DIR = os.path.join(ATTENTION_MOD_FMRI_DATA_DIR, "raw")

ATTENTION_MOD_FMRI_PREPROCESSED_DATA_DIR = os.path.join(ATTENTION_MOD_FMRI_DATA_DIR, "preprocessed")
ATTENTION_MOD_FMRI_PREPROCESSED_DATASINK_DIR = os.path.join(ATTENTION_MOD_FMRI_DATA_DIR, "datasink")
ATTENTION_MOD_FMRI_BETAS_DIR = os.path.join(ATTENTION_MOD_FMRI_DATA_DIR, "betas")
ATTENTION_MOD_FMRI_BETAS_SURFACE_DIR = os.path.join(ATTENTION_MOD_FMRI_DATA_DIR, "betas_surface_level")

ATTENTION_MOD_SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-05', 'sub-07']

RIDGE_DECODER_ATTN_MOD_OUT_DIR = os.path.join(ATTENTION_MOD_DATA_DIR, "whole_brain_decoding")


def nipype_subject_id(subject):
    return f'_subject_id_{subject}'


def model_features_file_path(model_name):
    return os.path.join(LATENT_FEATURES_DIR, f"{model_name.lower()}.p")


def append_images(images, horizontally=True, padding=5):
    if horizontally:
        append_axis = 0
        other_axis = 1
    else:
        append_axis = 1
        other_axis = 0

    imgs_dims = [0, 0]
    imgs_dims[append_axis] = np.sum([img.size[append_axis] for img in images]) + (len(images) - 1) * padding
    imgs_dims[other_axis] = np.max([img.size[other_axis] for img in images])
    full_img = Image.new("RGBA", (imgs_dims[0], imgs_dims[1]))

    prev_loc = [0, 0]
    for img in images:
        full_img.paste(img, (prev_loc[0], prev_loc[1]))
        prev_loc[append_axis] += img.size[append_axis] + padding

    return full_img


def save_plot_and_crop_img(path, crop_to_content=True, crop_cbar=False, horizontal_cbar=False):
    plt.savefig(path, dpi=300, transparent=True)
    image = Image.open(path)
    if crop_cbar:
        if horizontal_cbar:
            image = image.crop((0, int(image.size[1] - image.size[1] / 5), image.size[0], image.size[1]))
        else:
            image = image.crop((int(image.size[0] - image.size[0] / 5), 0, image.size[0], image.size[1]))
    if crop_to_content:
        image = image.crop(image.getbbox())
    image.save(path)
    plt.close()


def export_to_gifti(scores, path):
    data = scores.astype(np.float32)
    gimage = GiftiImage(
        darrays=[GiftiDataArray(
            data,
            intent=intent_codes.code['NIFTI_INTENT_NONE'],
            datatype=data_type_codes.code['NIFTI_TYPE_FLOAT32'])]
    )
    gimage.to_filename(path)
