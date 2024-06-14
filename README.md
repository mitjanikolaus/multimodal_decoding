# multimodal_decoding


## Visualization with freeview

for fsaverage5:
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

for fsaverage7:
```
freeview \
-f $FREESURFER_HOME/subjects/fsaverage/surf/lh.inflated\
:overlay=~/data/multimodal_decoding/searchlight/train/blip2/avg/fsaverage7/n_neighbors_200/p_values_metric_3_h_2.0_e_1.0_smoothed_0_lh.gii\
:annot=~/code/multimodal_decoding/multimodal_decoding/atlas_data/hcp_surface/lh.HCP-MMP1.annot\
:annot=$FREESURFER_HOME/subjects/fsaverage/label/lh.aparc.annot\
:annot=$FREESURFER_HOME/subjects/fsaverage/label/lh.aparc.a2009s.annot \
-f $FREESURFER_HOME/subjects/fsaverage/surf/rh.inflated\
:overlay=~/data/multimodal_decoding/searchlight/train/blip2/avg/fsaverage7/n_neighbors_200/p_values_metric_3_h_2.0_e_1.0_smoothed_0_rh.gii\
:annot=~/code/multimodal_decoding/multimodal_decoding/atlas_data/hcp_surface/rh.HCP-MMP1.annot\
:annot=$FREESURFER_HOME/subjects/fsaverage/label/rh.aparc.annot\
:annot=$FREESURFER_HOME/subjects/fsaverage/label/rh.aparc.a2009s.annot
```

# Appendix 

## fMRI Preprocessing

#### Requirements

- Parts of the preprocessing pipeline are based on SPM. Therefore, we require matlab and
[SPM version 12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) to be installed
(SPM should be installed to `~/apps/spm12/`).
- For conversion to MNI space, we rely on [freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall).


### (1) STC, Realignment and Coregistration

This script performs the following steps using SPM: 
1. Slice time correction (STC)
2. Realignment
3. Coregistration

```
python preprocessing/fmri_preprocessing.py
```

The input for this script are the raw fMRI BIDS found at `~/data/multimodal_decoding/fmri/raw/bids` as well as 
the corrected T1w images of the first session: `~/data/multimodal_decoding/fmri/raw/corrected_anat`.

### (2) Transformation to MNI space

First, we're running recon-all to generate cortical reconstructions for all subjects:
```
python preprocessing/recon_script.py
```

Then, we can convert all data to MNI space:
```
python preprocessing/transform_to_mni.py
```


### (3) Gray Matter Masks

Gray matter masks are used to perform the analysis only on voxels that belong to gray matter.
We consider a very inclusive mask, any voxel that has a probability greater than 0 to belong to gray matter tissue is
included. 

As a first step, we took the corrected T1w image of the first session for each subject
(e.g. `~/data/multimodal_decoding/fmri/raw/corrected_anat/sub-01/sub-01_ses-01_run-01_T1W.nii`) and segment it using
SPM. (The output of this step is part of the raw data folder, so you don't have to repeat it.)
Then, we take the c1 (gray matter) segmentation (e.g. `c1sub-01_ses-01_run-01_T1W.nii`) and use the following script to
create a binary mask.
```
python preprocessing/create_gray_matter_masks.py
```
Finally, the aforementioned script is also converting the mask to MNI space. The final masks are save to
`~/data/multimodal_decoding/fmri/preprocessed/graymatter_masks/sub-0*/mask.nii`.

### (4) Generation of beta values

We generate beta values for each stimulus (image or caption) using a GLM.

First create the matlab script, and then run it:
```
python preprocessing/split_glm_make_spm_level1design_job_mat_mni.py --stage 1
matlab -nodisplay  -r split_glm_spm_design.m,exit  -logfile matlab_output.txt

python preprocessing/split_glm_make_spm_level1design_job_mat_mni.py --stage 2
matlab -nodisplay  -r split_glm_spm_design_phase2,exit  -logfile matlab_output.txt
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