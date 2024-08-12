import nibabel as nib
from nibabel import freesurfer
from nilearn import datasets
from nilearn.surface import surface
import numpy as np
from utils import HEMIS
import seaborn as sns


if __name__ == "__main__":
    # labels: "/usr/local/freesurfer/7.4.1/FreeSurferColorLUT.txt"

    # mgz_orig = nib.load('/usr/local/freesurfer/7.4.1/subjects/fsaverage/mri/aparc.a2009s+aseg.mgz')
    mgz_orig = nib.load('/usr/local/freesurfer/7.4.1/subjects/fsaverage/mri/aparc+aseg.mgz')

    for hemi in HEMIS:
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage7")
        pial_mesh = fsaverage[f"pial_{hemi}"]

        surface_annot = surface.vol_to_surf(mgz_orig, pial_mesh, interpolation='nearest', n_samples=1).astype(np.int32)
        surface_annot[surface_annot > 85] = 0

        # surface_annot[surface_annot != 17] = 0
        print(np.unique(surface_annot))
        label_names = ['unk'] * (int(np.max(surface_annot)) + 1)
        # label_names = [f'label_{id}' for id in range(np.max(surface_annot)+1-len(np.unique(surface_annot)))]
        label_names = [f'label_{id}' for id in np.unique(surface_annot)] + label_names

        label_colors = [(color[0], color[1], color[2], 1, id) for id, color in enumerate(sns.color_palette(n_colors=np.max(surface_annot)+1))]

        label_colors = 255 * np.array(label_colors)

        freesurfer.write_annot(f'{hemi}_subcortical.annot', surface_annot, label_colors, label_names, fill_ctab =True)
