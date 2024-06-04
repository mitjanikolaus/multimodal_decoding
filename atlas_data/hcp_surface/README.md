# HCP converted atlas

Source: https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_fsaverage/3498446

## Conversion of annotations from fsaverage7 to fsaverage5

using freesurfer:
```
mri_surf2surf --srcsubject fsaverage --trgsubject fsaverage5 --hemi lh --sval-annot atlas_data/hcp_surface/lh.HCP-MMP1.annot --tval atlas_data/hcp_surface/lh.HCP-MMP1-fsaverage5.annot
mri_surf2surf --srcsubject fsaverage --trgsubject fsaverage5 --hemi rh --sval-annot atlas_data/hcp_surface/rh.HCP-MMP1.annot --tval atlas_data/hcp_surface/rh.HCP-MMP1-fsaverage5.annot
```