# multimodal_decoding

## Decoding analysis
```
python analyses/decoding/ridge_regression_decoding.py
```


## Searchlight analysis

Modality-agnostic decoder:
```
python analyses/searchlight/searchlight.py --model imagebind --n-neighbors 750 --n-jobs 15

```
Modality-specific decoders:
```
python analyses/searchlight/searchlight.py --model imagebind --n-neighbors 750 --training-mode images --n-jobs 15
python analyses/searchlight/searchlight.py --model imagebind --n-neighbors 750 --training-mode captions --n-jobs 15
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
- For conversion to surface space, we rely on [freesurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall).


### Downsampling of anatomical scan:
```
FSLOUTPUTTYPE='NIFTI' flirt.fsl -in ~/data/multimodal_decoding/fmri/raw/corrected_anat/sub-01/sub-01_ses-01_run-01_T1W.nii -ref ~/data/multimodal_decoding/fmri/raw/corrected_anat/sub-01/sub-01_ses-01_run-01_T1W.nii -applyisoxfm 2.0 -nosearch -out ~/data/multimodal_decoding/fmri/raw/corrected_anat/sub-01/sub-01_ses-01_run-01_T1W_downsampled_2mm.nii

```

This conversion decreases the voxel size of the anatomical scan from 1mm<sup>3</sup> to 2mm<sup>3</sup>. The functional
data (which has a voxel size of 3mm<sup>3</sup>) will be coregistered to this downsampled image. 


### (1) STC, Realignment and Coregistration

This script performs the following steps using SPM: 
1. Slice time correction (STC)
2. Realignment
3. Coregistration
4. Segmentation of the anatomical scan

```
python preprocessing/fmri_preprocessing.py
```

The input for this script are the raw fMRI BIDS found at `~/data/multimodal_decoding/fmri/raw/bids` as well as 
the corrected T1w images of the first session: `~/data/multimodal_decoding/fmri/raw/corrected_anat`.



### (3) Gray Matter Masks

Gray matter masks are used to perform the analysis only on voxels that belong to gray matter.
We consider a very inclusive mask, any voxel that has a probability greater than 0 to belong to gray matter tissue is
included. 

The input to this script is the c1 (gray matter) segmentation of the T1w image (anatomical scan) of the first session
(e.g. `~/data/multimodal_decoding/fmri/raw/corrected_anat/sub-01/c1sub-01_ses-01_run-01_T1W_downsampled_2mm.nii`), which
was created in the previous step.
The following script creates the gray matter masks for each subject.
```
python preprocessing/create_gray_matter_masks.py
```

### (4) Generation of beta values

We generate beta values for each stimulus (images and captions) using a GLM.
We first create the matlab scripts using python nipype scripts, and then run them:

```
python preprocessing/make_spm_design_job_mat.py --subjects sub-01
cd preprocessing && matlab -nodisplay -nosplash -nodesktop -r "run_spm_glm sub-01;exit;"  -logfile matlab_output.txt && cd -
```

__Note:__ Repeat these steps separately for each subject by adapting the subject identifier (`sub-01`) for both matlab
and python scripts!

#### Organization of beta values
Next, we can create symbolic links for all beta files that are organized into separate folders for
images/captions/imagery as well as train/test trials and contain the corresponding COCO ID in their name.

```
python preprocessing/create_symlinks_beta_files.py
```


### (5) Transformation to surface space

First, we're running recon-all to generate cortical reconstructions for all subjects (repeat for each subject):
```
python preprocessing/recon_script.py --subject sub-01
```

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

# fMRI preprocessing attention modulation condition

```
python preprocessing/fmri_preprocessing.py --subjects sub-01 --fmri-bids-dir ~/data/multimodal_decoding/attention_modulation/fmri/bids/ --out-data-dir ~/
data/multimodal_decoding/attention_modulation/fmri/preprocessed
python preprocessing/make_spm_design_job_mat_attention_mod.py --subjects sub-01
cd preprocessing && matlab -nodisplay -nosplash -nodesktop -r "run_spm_glm sub-01 unstructured_additional_test;exit;"  -logfile matlab_output.txt && cd -

python preprocessing/create_symlinks_beta_files.py --subjects sub-01 --unstructured-dir-name unstructured_additional_test --splits imagery_weak test_caption_attended test_caption_unattended test_image_attended test_image_unattended
python preprocessing/transform_to_surface.py --subjects sub-01 --splits imagery_weak test_caption_attended test_caption_unattended test_image_attended test_image_unattended
```