############################################
# Training multimodal linear brain decoders
# inputs can be of any modality
# outputs are uni-modal
############################################
import argparse
import time

import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn.decoding import SearchLight
from nilearn.decoding.searchlight import search_light
from nilearn.image import new_img_like, get_data
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_img
from nilearn.surface import surface
from sklearn import neighbors
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
import os
from glob import glob
import pickle

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, TEST_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS, get_nn_latent_data, \
    get_default_features, calculate_eval_metrics, pairwise_accuracy
from decoding_utils import get_distance_matrix
from tqdm import trange
import pandas as pd

from utils import IMAGERY_SCENES, TWO_STAGE_GLM_DATA_DIR, VISION_MEAN_FEAT_KEY

DEFAULT_N_JOBS = 5
NUM_CV_SPLITS = 3

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")


def get_graymatter_mask(subject):
    fmri_data_dir = os.path.join(TWO_STAGE_GLM_DATA_DIR, subject)
    gray_matter_mask_address = os.path.join(fmri_data_dir, f'unstructured', 'mask.nii')
    # gray_matter_mask_img = nib.load(gray_matter_mask_address)
    # gray_matter_mask_data = gray_matter_mask_img.get_fdata().astype(np.int32)
    # print(f"Gray matter mask size: {gray_matter_mask_data.sum()}")
    return gray_matter_mask_address

def get_fmri_data(subject, mode):
    imagery_scenes = IMAGERY_SCENES[subject]

    fmri_data_dir = os.path.join(TWO_STAGE_GLM_DATA_DIR, subject)
    fmri_addresses_regex = os.path.join(fmri_data_dir, f'betas_{mode}*', '*.nii')
    fmri_betas_addresses = np.array(sorted(glob(fmri_addresses_regex)))

    stim_ids = []
    stim_types = []
    for addr in fmri_betas_addresses:
        file_name = os.path.basename(addr)
        if 'I' in file_name:  # Image
            stim_id = int(file_name[file_name.find('I') + 1:-4])
            stim_types.append('image')
        elif 'C' in file_name:  # Caption
            stim_id = int(file_name[file_name.find('C') + 1:-4])
            stim_types.append('caption')
        else:  # imagery
            stim_id = int(file_name[file_name.find('.nii') - 1:-4])
            stim_id = imagery_scenes[stim_id - 1][1]
            stim_types.append('imagery')
        stim_ids.append(stim_id)

    stim_ids = np.array(stim_ids)
    stim_types = np.array(stim_types)

    return fmri_betas_addresses, stim_ids, stim_types


def run(args):
    for training_mode in args.training_modes:
        for subject in args.subjects:
            train_fmri, train_stim_ids, train_stim_types = get_fmri_data(subject, training_mode)
            test_fmri, test_stim_ids, test_stim_types = get_fmri_data(subject, args.testing_mode)
            gray_matter_mask = get_graymatter_mask(subject)

            for model_name in args.models:
                model_name = model_name.lower()

                for features in args.features:
                    if features == FEATS_SELECT_DEFAULT:
                        features = get_default_features(model_name)
                    print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                          f"MODEL: {model_name} | FEATURES: {features}")

                    train_data_latents, nn_latent_transform = get_nn_latent_data(model_name, features,
                                                                                 args.vision_features,
                                                                                 train_stim_ids,
                                                                                 subject,
                                                                                 training_mode)

                    if args.subset is not None:
                        train_fmri = train_fmri[:args.subset]
                        train_data_latents = train_data_latents[:args.subset]

                    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage3")
                    hemi = "left"
                    # Average voxels 5 mm close to the 3d pial surface
                    radius = 5.0
                    pial_mesh = fsaverage[f"pial_{hemi}"]
                    X = surface.vol_to_surf(train_fmri, pial_mesh, radius=radius, mask_img=gray_matter_mask).T
                    for x in X:
                        x[np.isnan(x)] = 0 # TODO
                    infl_mesh = fsaverage[f"infl_{hemi}"]
                    coords, _ = surface.load_surf_mesh(infl_mesh)
                    nn = neighbors.NearestNeighbors(radius=args.radius)
                    adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()
                    if args.subset is not None:
                        adjacency = adjacency[:args.subset]

                    model = make_pipeline(StandardScaler(), Ridge(alpha=args.l2_regularization_alpha))
                    pairwise_acc_scorer = make_scorer(pairwise_accuracy, greater_is_better=True)
                    cv = KFold(n_splits=NUM_CV_SPLITS)
                    # searchlight = SearchLight(mask_img=gray_matter_mask, process_mask_img=adjacency,
                    #                           radius=args.radius, estimator=model,
                    #                           n_jobs=args.n_jobs, scoring=pairwise_acc_scorer, cv=cv,
                    #                           verbose=3)

                    start = time.time()
                    scores = search_light(X, train_data_latents, estimator=model, A=adjacency, cv=cv, n_jobs=args.n_jobs,
                                          scoring=pairwise_acc_scorer, verbose=3)

                    # searchlight.fit(X, train_data_latents)
                    end = time.time()
                    print(f"Elapsed time: {int(end - start)}s")

                    results_dir = os.path.join(SEARCHLIGHT_OUT_DIR, training_mode, subject)
                    os.makedirs(results_dir, exist_ok=True)

                    pickle.dump(scores, open(os.path.join(results_dir, "searchlight_test.p"), 'wb'))

                    # searchlight_img = new_img_like(gray_matter_mask, searchlight.scores_)
                    #
                    # plot_img(
                    #     searchlight_img,
                    #     bg_img=gray_matter_mask, # TODO mean
                    #     title="Searchlight",
                    #     display_mode="z",
                    #     cut_coords=[-9],
                    #     vmin=0.42,
                    #     cmap="hot",
                    #     threshold=0.2,
                    #     black_bg=True,
                    # )
                    #
                    # nifti_masker = NiftiMasker(
                    #     mask_img=gray_matter_mask,
                    #     runs=run,
                    #     standardize="zscore_sample",
                    #     memory="nilearn_cache",
                    #     memory_level=1,
                    # )
                    # fmri_masked = nifti_masker.fit_transform(fmri_img)
                    #
                    # from sklearn.feature_selection import f_classif
                    #
                    # _, p_values = f_classif(fmri_masked, y)
                    # p_values = -np.log10(p_values)
                    # p_values[p_values > 10] = 10
                    # p_unmasked = get_data(nifti_masker.inverse_transform(p_values))
                    #
                    # # F_score results
                    # p_ma = np.ma.array(p_unmasked, mask=np.logical_not(process_mask))
                    # f_score_img = new_img_like(mean_fmri, p_ma)
                    # plot_stat_map(
                    #     f_score_img,
                    #     mean_fmri,
                    #     title="F-scores",
                    #     display_mode="z",
                    #     cut_coords=[-9],
                    #     colorbar=False,
                    # )

                    # best_alpha = searchlight.best_params_["alpha"]

                    # results = {
                    #     "alpha": best_alpha,
                    #     "model": model_name,
                    #     "subject": subject,
                    #     "features": features,
                    #     "vision_features": args.vision_features,
                    #     "training_mode": training_mode,
                    #     "testing_mode": args.testing_mode,
                    #     "mask": mask,
                    #     "num_voxels": test_fmri_betas.shape[1],
                    #     "best_val_acc": True,
                    #     "cv_results": clf.cv_results_
                    # }
                    #
                    # test_data_latents, _ = get_nn_latent_data(model_name, features, args.vision_features,
                    #                                           test_stim_ids,
                    #                                           subject,
                    #                                           args.testing_mode,
                    #                                           nn_latent_transform=nn_latent_transform)
                    # best_model = clf.best_estimator_
                    # test_predicted_latents = best_model.predict(test_fmri_betas)
                    #
                    # test_results = {"stimulus_ids": test_stim_ids,
                    #                 "stimulus_types": test_stim_types,
                    #                 "predictions": test_predicted_latents,
                    #                 "latents": test_data_latents}
                    # test_results = calculate_eval_metrics(test_results, test_fmri_betas)
                    # print(f"Best alpha: {best_alpha} | Pairwise acc: {test_results['acc_cosine']:.2f}"
                    #       f" | Pairwise acc (captions): {test_results['acc_cosine_captions']:.2f}"
                    #       f" | Pairwise acc (images): {test_results['acc_cosine_images']:.2f}")
                    # # f" | RSA (captions): {test_results['rsa_captions']:.2f}"
                    # # f" | RSA (images): {test_results['rsa_images']:.2f}")
                    #
                    # results = results | test_results
                    #
                    # results_dir = os.path.join(SEARCHLIGHT_OUT_DIR, training_mode, subject)
                    # run_str = get_run_str(model_name, features, mask, best_val_acc=True)
                    # results_file_dir = f'{results_dir}/{run_str}'
                    # os.makedirs(results_file_dir, exist_ok=True)
                    #
                    # pickle.dump(results, open(os.path.join(results_file_dir, "results.p"), 'wb'))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)
    parser.add_argument("--testing-mode", type=str, default='test', choices=TEST_MODE_CHOICES)

    parser.add_argument("--subset", type=int, default=None)

    parser.add_argument("--models", type=str, nargs='+', default=['clip'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=VISION_MEAN_FEAT_KEY,
                        choices=VISION_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--l2-regularization-alpha", type=float, default=1e3)

    parser.add_argument("--radius", type=float, default=2)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(SEARCHLIGHT_OUT_DIR, exist_ok=True)

    run(args)
