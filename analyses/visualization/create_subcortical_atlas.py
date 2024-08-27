import nibabel as nib
from nibabel import freesurfer
from nibabel.freesurfer.io import _pack_rgb
from nilearn import datasets
from nilearn.surface import surface
import numpy as np
from utils import HEMIS, FREESURFER_HOME_DIR, ROOT_DIR
import seaborn as sns


if __name__ == "__main__":
    # labels: "/usr/local/freesurfer/7.4.1/FreeSurferColorLUT.txt"

    mgz_orig = nib.load(f'{FREESURFER_HOME_DIR}/subjects/fsaverage/mri/aparc.a2009s+aseg.mgz')
    # mgz_orig = nib.load(f'{FREESURFER_HOME_DIR}/subjects/fsaverage/mri/aparc+aseg.mgz')

    label_names_info = dict()
    with open(f'/{FREESURFER_HOME_DIR}/FreeSurferColorLUT.txt') as file:
        for line in file.read().split("\n"):
            if len(line) > 0 and line[0] != '#':
                line_elements = [el for el in line.split(' ') if len(el) > 0]
                label_names_info[int(line_elements[0])] = line_elements[1]

    for hemi in HEMIS:
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage7")
        pial_mesh = fsaverage[f"pial_{hemi}"]

        surface_annot = surface.vol_to_surf(mgz_orig, pial_mesh, interpolation='nearest', n_samples=1).astype(np.int32)
        surface_annot[surface_annot > 85] = 0 #TODO right hemi
        surface_annot[surface_annot == 0] = -1

        label_ids = np.unique(surface_annot)
        print(label_ids)

        label_to_id = {id: id if id == -1 else i for i, id in enumerate(label_ids)}

        label_names_info = {label_to_id[l] if l in label_to_id else -1: name for l, name in label_names_info.items()}
        # id_to_label = {val: key for key, val in label_to_id.items()}

        surface_annot_transformed = np.array([label_to_id[l] for l in surface_annot])

        # label_names = ['unk'] * (int(np.max(surface_annot)) + 1)
        label_names = [label_names_info[id] if id in label_names_info else "unk" for id in range(int(np.max(surface_annot_transformed))+1)]
        # label_names = [label_names_info[id] if id in label_names_info else 'unk' for id in range(int(np.max(surface_annot))+1)]
        print(label_names)

        # colors = np.array(sns.color_palette(n_colors=np.max(surface_annot)+1)) * 255
        colors = np.array(sns.color_palette(n_colors=np.max(surface_annot_transformed)+1)) * 255
        colors = np.array([(color[0], color[1], color[2], 255) for id, color in enumerate(colors)])

        # label_colors_map = [(colors[id][0], colors[id][1], colors[id][2], 255, id) if id in colors else (0, 0, 0, 0, id) for id in range(len(label_names))]

        # label_colors_map = np.array(label_colors_map)

        # label_colors_map[:, -1][surface_annot_transformed]
        # np.array_equal(label_colors_map[:, [4]], _pack_rgb(label_colors_map[:, :3]))
        freesurfer.write_annot(f'{ROOT_DIR}/atlas_data/{hemi}_subcortical.annot', surface_annot_transformed, colors, label_names, fill_ctab=True)
