import argparse

import numpy as np
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.surface import surface

import os
import pickle

from tqdm import tqdm

from data import get_fmri_data_paths, INDICES_TEST_STIM_IMAGE, TEST_STIM_IDS, INDICES_TEST_STIM_CAPTION, IMAGERY_SCENES, \
    SPLIT_IMAGERY, SPLIT_TRAIN, SPLIT_TEST, TEST_STIM_TYPES, IMAGERY_STIMS_IDS, IMAGERY_STIMS_TYPES, \
    get_fmri_voxel_data, IMAGE, CAPTION
from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import FMRI_BETAS_SURFACE_DIR, SUBJECTS, DEFAULT_RESOLUTION, FMRI_BETAS_DIR, FMRI_STIM_INFO_DIR, \
    export_to_gifti, FS_HEMI_NAMES, FREESURFER_SUBJECTS_DIR


def run(args):
    os.environ["SUBJECTS_DIR"] = FREESURFER_SUBJECTS_DIR

    for subject in args.subjects:
        print("\n", subject)
        test_fmri, test_stim_ids, test_stim_types = get_fmri_data_paths(args.betas_dir, subject, SPLIT_TEST)
        imagery_fmri, imagery_stim_ids, imagery_stim_types = get_fmri_data_paths(args.betas_dir, subject, SPLIT_IMAGERY)

        assert np.all(test_stim_types[INDICES_TEST_STIM_IMAGE] == IMAGE)
        assert np.all(test_stim_types[INDICES_TEST_STIM_CAPTION] == CAPTION)
        assert np.all(test_stim_ids == TEST_STIM_IDS)
        assert np.all(test_stim_types == TEST_STIM_TYPES)
        assert np.all(imagery_stim_ids == IMAGERY_STIMS_IDS[subject])
        assert np.all(imagery_stim_types == IMAGERY_STIMS_TYPES[subject])
        assert np.all(imagery_stim_ids == [i[1] for i in IMAGERY_SCENES[subject]])

        train_fmri, _, _ = get_fmri_data_paths(args.betas_dir, subject, SPLIT_TRAIN)

        jobs = []
        for hemi in args.hemis:
            for path in tqdm(train_fmri + test_fmri + imagery_fmri):
                path_out = path.replace(args.betas_dir, os.path.join(args.betas_dir, 'surface'))
                path_out = path_out.replace('.nii', '.gii')
                assert path != path_out

                os.makedirs(os.path.dirname(path_out), exist_ok=True)
                cmd = (f"mri_vol2surf --mov {path} --o {path_out} --hemi {FS_HEMI_NAMES[hemi]} --trgsubject fsaverage "
                       f"--regheader {subject} --interp trilinear --projfrac 0.5")
                jobs.append(cmd)

        def exec_command(cmd, silent=True):
            if silent:
                cmd = cmd + " >/dev/null 2>&1"
            result_code = os.system(cmd)
            if result_code != 0:
                raise RuntimeError(f"failed to convert {path} to surface {result_code}")

        Parallel(n_jobs=int(args.n_jobs))(
            delayed(exec_command)(cmd)
            for cmd in tqdm(jobs)
        )

        # gray_matter_mask = get_graymatter_mask_path(subject)
        #
        # fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
        # for hemi in args.hemis:
        #     print("Hemisphere: ", hemi)
        #     pial_mesh = fsaverage[f"pial_{hemi}"]
        #     white_matter_mesh = fsaverage[f"white_{hemi}"]
        #
        #     print("transforming to surface.. (test)", end=" ")
        #     surface_projection = surface.vol_to_surf(test_fmri, pial_mesh, mask_img=gray_matter_mask, inner_mesh=white_matter_mesh, interpolation='linear').T
        #     print("done.")
        #     results_file_name = f"{subject}_{hemi}_{args.resolution}_test.p"
        #     pickle.dump(surface_projection, open(os.path.join(args.out_dir, results_file_name), 'wb'))
        #     print("saved.")
        #
        #     test_fmri_betas_volume_space, _, _ = get_fmri_voxel_data(args.betas_dir, subject, SPLIT_TEST)
        #     print(f'original nans: {np.sum(np.isnan(test_fmri_betas_volume_space[0]))} ({100*np.mean(np.isnan(test_fmri_betas_volume_space[0])):.2f}%)')
        #     print(f'transformed nans: {np.sum(np.isnan(surface_projection[0]))} ({100*np.mean(np.isnan(surface_projection[0])):.2f}%)')
        #
        #     print("transforming to surface.. (imagery)", end=" ")
        #     surface_projection = surface.vol_to_surf(imagery_fmri, pial_mesh, mask_img=gray_matter_mask, inner_mesh=white_matter_mesh, interpolation='linear').T
        #     print("done.")
        #     results_file_name = f"{subject}_{hemi}_{args.resolution}_imagery.p"
        #     pickle.dump(surface_projection, open(os.path.join(args.out_dir, results_file_name), 'wb'))
        #     print("saved.")
        #
        # train_fmri, train_stim_ids, train_stim_types = get_fmri_data_paths(args.betas_dir, subject, SPLIT_TRAIN)
        # pickle.dump(train_stim_ids, open(os.path.join(FMRI_STIM_INFO_DIR, f"{subject}_stim_ids_train.p"), 'wb'))
        # pickle.dump(train_stim_types, open(os.path.join(FMRI_STIM_INFO_DIR, f"{subject}_stim_types_train.p"), 'wb'))
        #
        # for hemi in args.hemis:
        #     print("Hemisphere: ", hemi)
        #     pial_mesh = fsaverage[f"pial_{hemi}"]
        #     white_matter_mesh = fsaverage[f"white_{hemi}"]
        #
        #     print("transforming to surface.. (train part 1)", end=" ")
        #     surface_projection = surface.vol_to_surf(train_fmri[:2500], pial_mesh, mask_img=gray_matter_mask, inner_mesh=white_matter_mesh, interpolation='linear').T
        #     print("done.")
        #     results_file_name_1 = f"{subject}_{hemi}_train_1.p"
        #     pickle.dump(surface_projection, open(os.path.join(args.out_dir, results_file_name_1), 'wb'))
        #     print("saved.")
        #
        #     print("transforming to surface.. (train part 2)", end=" ")
        #     pial_mesh = fsaverage[f"pial_{hemi}"]
        #     surface_projection = surface.vol_to_surf(train_fmri[2500:5000], pial_mesh, mask_img=gray_matter_mask, inner_mesh=white_matter_mesh, interpolation='linear').T
        #     print("done.")
        #     results_file_name_2 = f"{subject}_{hemi}_train_2.p"
        #     pickle.dump(surface_projection, open(os.path.join(args.out_dir, results_file_name_2), 'wb'))
        #     print("saved.")
        #
        #     print("transforming to surface.. (train part 3)", end=" ")
        #     pial_mesh = fsaverage[f"pial_{hemi}"]
        #     surface_projection = surface.vol_to_surf(train_fmri[5000:7500], pial_mesh, mask_img=gray_matter_mask, inner_mesh=white_matter_mesh, interpolation='linear').T
        #     print("done.")
        #     results_file_name_3 = f"{subject}_{hemi}_train_3.p"
        #     pickle.dump(surface_projection, open(os.path.join(args.out_dir, results_file_name_3), 'wb'))
        #     print("saved.")
        #
        #     print("transforming to surface.. (train part 4)", end=" ")
        #     pial_mesh = fsaverage[f"pial_{hemi}"]
        #     surface_projection = surface.vol_to_surf(train_fmri[7500:], pial_mesh, mask_img=gray_matter_mask, inner_mesh=white_matter_mesh, interpolation='linear').T
        #     print("done.")
        #     results_file_name_4 = f"{subject}_{hemi}_train_4.p"
        #     pickle.dump(surface_projection, open(os.path.join(args.out_dir, results_file_name_4), 'wb'))
        #     print("saved.")
        #
        #     X_1 = pickle.load(open(os.path.join(args.out_dir, results_file_name_1), 'rb'))
        #     X_2 = pickle.load(open(os.path.join(args.out_dir, results_file_name_2), 'rb'))
        #     X_3 = pickle.load(open(os.path.join(args.out_dir, results_file_name_3), 'rb'))
        #     X_4 = pickle.load(open(os.path.join(args.out_dir, results_file_name_4), 'rb'))
        #     results_file_name = f"{subject}_{hemi}_{args.resolution}_train.p"
        #     pickle.dump(np.concatenate((X_1, X_2, X_3, X_4)), open(os.path.join(args.out_dir, results_file_name), 'wb'))
        #     os.remove(os.path.join(args.out_dir, results_file_name_1))
        #     os.remove(os.path.join(args.out_dir, results_file_name_2))
        #     os.remove(os.path.join(args.out_dir, results_file_name_3))
        #     os.remove(os.path.join(args.out_dir, results_file_name_4))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    parser.add_argument("--n-jobs", type=str, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
