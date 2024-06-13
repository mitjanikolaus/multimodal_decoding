# multimodal_decoding


## Visualization with freeview

```
freeview \
-f $FREESURFER_HOME/subjects/fsaverage5/surf/lh.inflated\
:overlay=~/data/multimodal_decoding/searchlight/train/blip2/avg/fsaverage5/n_neighbors_200/p_values_metric_3_h_2.0_e_1.0_smoothed_0_lh.gii\
:annot=~/code/multimodal_decoding/multimodal_decoding/atlas_data/hcp_surface/lh.HCP-MMP1-fsaverage5.annot\
:annot=$FREESURFER_HOME/subjects/fsaverage5/label/lh.aparc.annot\
:annot=$FREESURFER_HOME/subjects/fsaverage5/label/lh.aparc.a2009s.annot \
-f $FREESURFER_HOME/subjects/fsaverage5/surf/rh.inflated\
:overlay=~/data/multimodal_decoding/searchlight/train/blip2/avg/fsaverage5/n_neighbors_200/p_values_metric_3_h_2.0_e_1.0_smoothed_0_rh.gii\
:annot=~/code/multimodal_decoding/multimodal_decoding/atlas_data/hcp_surface/rh.HCP-MMP1-fsaverage5.annot\
:annot=$FREESURFER_HOME/subjects/fsaverage5/label/rh.aparc.annot\
:annot=$FREESURFER_HOME/subjects/fsaverage5/label/rh.aparc.a2009s.annot
```

# Appendix 

## fMRI Preprocessing

### STC, Realignment and Coregistration

This script performs the following steps using SPM: 
1. Slice time correction (STC)
2. Realignment
3. Coregistration

```
python preprocessing/fmri_preprocessing.py
```

#### Requirements

This script requires matlab and SPM version 12 (installed at `~/apps/spm12/`).

#### Input data:
- fmri BIDS: `~/data/multimodal_decoding/fmri/raw/bids`
- corrected T1w: `~/data/multimodal_decoding/fmri/raw/corrected_anat`

#### Output data:
- preprocessed data: `~/data/multimodal_decoding/fmri/preprocessed/datasink`


### Transformation to MNI space

```
python preprocessing/raw_data_to_mni.py
```

#### Requirements

For conversion to MNI space, this script requires freesurfer to be installed.


### Gray Matter Mask

Gray matter masks are used to perform the analysis only on voxels that belong to gray matter.
We consider a very inclusive mask, any voxel that has a probability greater than 0 to belong to gray matter tissue is
included. The script creates a mask for each subject and converts it to MNI space.

```
python preprocessing/create_gray_matter_masks.py
```


## DNN Feature extraction 

### VisualBERT feature extraction

```
pip install opencv-python pyyaml==5.1
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### LXMERT features

Precomputed bottom-up features can be downloaded from [this link](https://storage.googleapis.com/up-down-attention/trainval.zip)
(Source: https://github.com/peteanderson80/bottom-up-attention).

The downloaded features need to be extracted to `~/data/coco/bottom_up_feats/`.