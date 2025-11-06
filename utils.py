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

RESULTS_FILE = "results.csv"
PREDICTIONS_FILE = "predictions.p"

FMRI_DATA_DIR = os.path.join(DATA_DIR, "fmri")
FMRI_STIM_INFO_DIR = os.path.join(FMRI_DATA_DIR, "stim_info")
FMRI_DOWNSAMPLED_ANAT_DATA_DIR = os.path.join(FMRI_DATA_DIR, "anat_downsampled")
FMRI_NORMALIZATIONS_DIR = os.path.join(FMRI_DATA_DIR, "normalizations")
FMRI_PREPROCESSED_DATA_DIR = os.path.join(FMRI_DATA_DIR, "preprocessed")
FMRI_PREPROCESSING_DATASINK_DIR = os.path.join(FMRI_PREPROCESSED_DATA_DIR, "datasink")

FMRI_BIDS_DATA_DIR = os.path.join(FMRI_DATA_DIR, "bids")

STIM_INFO_PATH = os.path.join(FMRI_BIDS_DATA_DIR, "stimuli", "stimuli.csv")
FMRI_BETAS_DIR = os.path.join(FMRI_BIDS_DATA_DIR, "derivatives", "betas")

FMRI_BETAS_SURFACE_DIR = os.path.join(FMRI_BETAS_DIR, "surface")

FREESURFER_BASE_DIR = os.path.join(DATA_DIR, "freesurfer")
FREESURFER_SUBJECTS_DIR = os.path.join(FREESURFER_BASE_DIR, "subjects_downsampled_2mm")

FREESURFER_HOME_DIR = "/usr/local/freesurfer/7.4.1"

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
RIDGE_DECODER_OUT_DIR = os.path.join(DATA_DIR, "whole_brain_decoding")

SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']
HEMIS = ['left', 'right']
HEMIS_FS = ['lh', 'rh']

FS_HEMI_NAMES = {'left': 'lh', 'right': 'rh'}
FS_NUM_VERTICES = 163842

DEFAULT_MODEL = "imagebind"
DEFAULT_RESOLUTION = "fsaverage7"

METRIC_WITHIN_MODALITY_DECODING = 'within'
METRIC_CROSS_DECODING = 'cross'

METRIC_WITHIN_MODALITY_DECODING_WITH_ATTENTION_TO_OTHER_MOD = 'within_with_attention_to_other_mod'
METRIC_WITHIN_MODALITY_DECODING_WITH_ATTENTION_TO_OTHER_MOD_IMAGES = 'within_with_attention_to_other_mod_images'
METRIC_WITHIN_MODALITY_DECODING_WITH_ATTENTION_TO_OTHER_MOD_CAPTIONS = 'within_with_attention_to_other_mod_captions'

METRIC_CROSS_DECODING_WITH_ATTENTION_TO_STIMULUS_MOD = 'cross_with_attention_to_stimulus_mod'
METRIC_CROSS_DECODING_WITH_ATTENTION_TO_OTHER_MOD = 'cross_with_attention_to_other_mod'

METRIC_DIFF_ATTENTION_WITHIN_MODALITY = 'diff_attention_cross'
METRIC_DIFF_ATTENTION_CROSS_MODALITY = 'diff_attention_within'

METRIC_DIFF_ATTEND_BOTH_VS_OTHER_WITHIN_MODALITY = 'diff_attend_both_vs_other_cross'
METRIC_DIFF_ATTEND_BOTH_VS_OTHER_CROSS_MODALITY = 'diff_attend_both_vs_other_within'

METRIC_GW = 'gw'
METRIC_MOD_INVARIANT_ATTENDED = 'mod_invariant_attended'
METRIC_MOD_INVARIANT_UNATTENDED = 'mod_invariant_unattended'
METRIC_MOD_INVARIANT_ATTENDED_ALT = 'mod_invariant_attended_alt'
METRIC_MOD_INVARIANT_UNATTENDED_ALT = 'mod_invariant_unattended_alt'
METRIC_ATTENTION_DIFF_IMAGES = "attn_diff_images"
METRIC_ATTENTION_DIFF_CAPTIONS = "attn_diff_captions"
METRIC_MOD_INVARIANT_INCREASE = 'mod_invariant_increase'

DIFF = "diff"
DIFF_DECODERS = "diffdecoders"

ADDITIONAL_TEST_DATA_DIR = os.path.join(DATA_DIR, "additional_test")
ADDITIONAL_TEST_FMRI_DATA_DIR = os.path.join(ADDITIONAL_TEST_DATA_DIR, "fmri")
ADDITIONAL_TEST_FMRI_RAW_BIDS_DATA_DIR = os.path.join(ADDITIONAL_TEST_FMRI_DATA_DIR, "bids")

ADDITIONAL_TEST_FMRI_PREPROCESSED_DATA_DIR = os.path.join(ADDITIONAL_TEST_FMRI_DATA_DIR, "preprocessed")
ADDITIONAL_TEST_FMRI_PREPROCESSED_DATASINK_DIR = os.path.join(ADDITIONAL_TEST_FMRI_PREPROCESSED_DATA_DIR, "datasink")
ADDITIONAL_TEST_UNSTRUCTURED_DIR_NAME = "unstructured_additional_test"

SUBJECTS_ADDITIONAL_TEST = ['sub-01', 'sub-02', 'sub-04', 'sub-05', 'sub-07']

DECODER_ADDITIONAL_TEST_OUT_DIR = os.path.join(ADDITIONAL_TEST_DATA_DIR, "whole_brain_decoding")


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
            # image = image
            image = image.crop((0, 0, image.size[0], (int(image.size[1] - image.size[1] / 5))))
            # image = image.crop((int(image.size[0] - image.size[0] / 5), 0, image.size[0], image.size[1]))
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
