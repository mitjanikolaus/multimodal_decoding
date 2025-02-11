# multimodal_decoding

## Decoding analysis
```
CUDA_AVAILABLE_DEVICE=0 python analyses/decoding/ridge_regression_decoding.py --cuda
```



## Encoding analysis

```
CUDA_VISIBLE_DEVICES=0 python analyses/encoding/ridge_regression_encoding.py --models imagebind --training-modes train_image train_caption --cuda --create-null-distr
```

## Searchlight analysis

Modality-agnostic decoder:
```
python analyses/searchlight/searchlight.py --model imagebind --n-neighbors 750 --n-jobs 15

```
Modality-specific decoders:
```
python analyses/searchlight/searchlight.py --model imagebind --n-neighbors 750 --features vision --test-features vision --training-mode train_images --n-jobs 15
python analyses/searchlight/searchlight.py --model imagebind --n-neighbors 750 --features lang --test-features lang --training-mode train_captions --n-jobs 15
```
## Visualization with freeview


```
python analyses/view_results_freeview.py 
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

Then, we create an LTA (Linear Transform Archive) file for conversion of functional scans from the subject space to MNI
space (repeat this for all subjects).
```
tkregisterfv --mov ~/data/multimodal_decoding/fmri/preprocessed/preprocess_workflow/_subject_id_sub-01/_session_id_ses-01/coregister/rameanasub-01_ses-01_task-coco_run-01_bold.nii --s sub-01 --regheader --reg ~/data/multimodal_decoding/freesurfer/regfiles/sub-01/spm2fs
```


Finally, we can convert all data to MNI space:
```
python preprocessing/transform_to_mni.py
```

Note: The mni conversion also increases the voxel size from 1mm<sup>3</sup> to 2mm<sup>3</sup>.

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
Finally, the aforementioned script is also converting the mask to MNI space. The final masks are saved to
`~/data/multimodal_decoding/fmri/preprocessed/graymatter_masks/sub-0*/mask_mni.nii`.

### (4) Generation of beta values

We generate beta values for each stimulus (images and captions) using a GLM.
We first create the matlab scripts using python nipype scripts, and then run them:

```
python preprocessing/make_spm_design_job_mat.py --subjects sub-01
cd preprocessing && matlab -nodisplay -nosplash -nodesktop -r "run_spm_glm sub-01;exit;"  -logfile matlab_output.txt && cd -
```

__Note:__ Repeat these steps separately for each subject by adapting the subject identifier (`sub-01`) for both matlab and python
scripts!

#### Organization of beta values
Next, we can create symbolic links for all beta files that are organized into separate folders for
images/captions/imagery as well as train/test trials and contain the corresponding COCO ID in their name.

```
python preprocessing/create_symlinks_beta_files.py
```

### (5) Transformation to surface space

Then, we can convert all data to surface space to perform the searchlight analyses:

```
python preprocessing/transform_to_surface.py
```


## DNN Feature extraction 

### BLIP2

Follow instructions on [https://github.com/salesforce/LAVIS/](https://github.com/salesforce/LAVIS/) to create a python
environment.

### VisualBERT

```
pip install opencv-python pyyaml==5.1
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### LXMERT

Precomputed bottom-up features can be downloaded from [this link](https://storage.googleapis.com/up-down-attention/trainval.zip)
(Source: https://github.com/peteanderson80/bottom-up-attention).

The downloaded features need to be extracted to `~/data/coco/bottom_up_feats/`.