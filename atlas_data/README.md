# Creation of atlas for hippocampus

```
mri_vol2surf --mov /usr/local/freesurfer/7.4.1/subjects/fsaverage/mri/aparc.a2009s+aseg.mgz --regheader fsaverage --o test_atlas_lh.mgh --hemi lh --trgsubject fsaverage --projfrac 0.5
mris_seg2annot --seg test_atlas_lh.mgh  --h lh --s fsaverage --ctab /usr/local/freesurfer/7.4.1/FreeSurferColorLUT.txt --o ./lh.test.annot
```