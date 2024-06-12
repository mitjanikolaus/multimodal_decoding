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


### Gray Matter Mask

A mask is used to perform the analysis only on gray matter voxels. 

These masks are created using the matlab library imcalc:

- download imcalc from [http://tools.robjellis.net/](http://tools.robjellis.net/) and
extract it to `~/apps/spm12/toolbox/imcalc/`
- then, start matlab and run:

```
addpath ~/apps/spm12
addpath(genpath('~/apps/spm12/toolbox/imcalc'))
imcalc
```

### Transformation to MNI space

```
python preprocessing/raw_data_to_mni.py
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