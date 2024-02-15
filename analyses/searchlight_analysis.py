import argparse
import numpy as np
from matplotlib.patches import Circle
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle

from nilearn.image import resample_to_img
from nilearn.surface import surface
from tqdm import tqdm

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS

from utils import VISION_MEAN_FEAT_KEY, RESULTS_DIR

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")

COLORBAR_MAX = 0.85
COLORBAR_THRESHOLD_MIN = 0.6
COLORBAR_DIFFERENCE_THRESHOLD_MIN = 0.1
VIEWS = ["lateral", "medial"]#, "ventral"]   #, "ventral"]


def run(args):
    all_scores = []

    results_regex = os.path.join(SEARCHLIGHT_OUT_DIR, f'train/*/*/*/fsaverage6/left/*/alpha*.p')
    results_paths = np.array(sorted(glob(results_regex)))
    for path in results_paths:
        training_mode = os.path.dirname(path).split("/")[-7]
        mode = os.path.dirname(path).split("/")[-1]
        resolution = os.path.dirname(path).split("/")[-3]
        subject = os.path.dirname(path).split("/")[-4]
        model_name = os.path.dirname(path).split("/")[-6]
        alpha = float(os.path.basename(path).split("_")[1][:-2])

        scores = dict()
        print(path)
        for hemi in ['left', 'right']:

            scores[hemi] = dict()
            path_scores_hemi = path.replace('left', hemi)
            if os.path.isfile(path_scores_hemi):
                scores_hemi = pickle.load(open(path_scores_hemi, 'rb'))['scores']
                for testing_mode in ["test_overall", "test_captions", "test_images"]:
                    score_name = "overall" if testing_mode == "test_overall" else testing_mode
                    scores[hemi][score_name] = np.array([score[testing_mode] for score in scores_hemi])

                print(hemi, {n: round(score.mean(), 4) for n, score in scores[hemi].items()})
                print(hemi, {f"{n}_max": round(score.max(), 2) for n, score in scores[hemi].items()})
                scores[hemi]["min(captions,images)"] = np.min((scores[hemi]['test_images'], scores[hemi]['test_captions']), axis=0)

                path_scores_hemi_captions = path.replace('train/', 'train_captions/')
                scores_mod_specific_captions = dict()
                if os.path.isfile(path_scores_hemi_captions):
                    scores_hemi_captions = pickle.load(open(path_scores_hemi_captions, 'rb'))['scores']
                    for testing_mode in ["test_overall", "test_captions", "test_images"]:
                        score_name = "overall" if testing_mode == "test_overall" else testing_mode
                        scores_mod_specific_captions[score_name] = np.array([score[testing_mode] for score in scores_hemi_captions])

                path_scores_hemi_images = path.replace('train/', 'train_images/')
                scores_mod_specific_images = dict()
                if os.path.isfile(path_scores_hemi_images):
                    scores_hemi_images = pickle.load(open(path_scores_hemi_images, 'rb'))['scores']
                    for testing_mode in ["test_overall", "test_captions", "test_images"]:
                        score_name = "overall" if testing_mode == "test_overall" else testing_mode
                        scores_mod_specific_images[score_name] = np.array([score[testing_mode] for score in scores_hemi_images])

                if len(scores_mod_specific_captions) > 0 and len(scores_mod_specific_images) > 0:
                    scores[hemi]['mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific)'] = np.array([np.mean((ai, ac)) - np.mean((si, sc)) for ai, ac, si, sc in zip(scores[hemi]['test_images'], scores[hemi]['test_captions'], scores_mod_specific_images['test_images'], scores_mod_specific_captions['test_captions'])])
                    scores[hemi]['imgs_agno - imgs_specific'] = np.array([ai - si for ai, ac, si, sc in zip(scores[hemi]['test_images'], scores[hemi]['test_captions'], scores_mod_specific_images['test_images'], scores_mod_specific_captions['test_captions'])])
                    scores[hemi]['captions_agno - captions_specific'] = np.array([ac - sc for ai, ac, si, sc in zip(scores[hemi]['test_images'], scores[hemi]['test_captions'], scores_mod_specific_images['test_images'], scores_mod_specific_captions['test_captions'])])

                    scores[hemi]['imgs_agno - imgs_specific (cross)'] = np.array([ai - si for ai, ac, si, sc in
                                                                              zip(scores[hemi]['test_images'],
                                                                                  scores[hemi]['test_captions'],
                                                                                  scores_mod_specific_captions[
                                                                                      'test_images'],
                                                                                  scores_mod_specific_images[
                                                                                      'test_captions'])])
                    scores[hemi]['captions_agno - captions_specific (cross)'] = np.array([ac - sc for ai, ac, si, sc in
                                                                                  zip(scores[hemi]['test_images'],
                                                                                      scores[hemi]['test_captions'],
                                                                                      scores_mod_specific_captions[
                                                                                          'test_images'],
                                                                                      scores_mod_specific_images[
                                                                                          'test_captions'])])
                    scores[hemi]['mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific) (cross)'] = np.array([np.mean((ai, ac)) - np.mean((si, sc)) for ai, ac, si, sc in zip(scores[hemi]['test_images'], scores[hemi]['test_captions'], scores_mod_specific_captions['test_images'], scores_mod_specific_images['test_captions'])])
                    scores[hemi]['imgs_specific (cross)'] = np.array([si for ai, ac, si, sc in
                                                                                  zip(scores[hemi]['test_images'],
                                                                                      scores[hemi]['test_captions'],
                                                                                      scores_mod_specific_captions[
                                                                                          'test_images'],
                                                                                      scores_mod_specific_images[
                                                                                          'test_captions'])])
                    scores[hemi]['captions_specific (cross)'] = np.array([sc for ai, ac, si, sc in
                                                                                  zip(scores[hemi]['test_images'],
                                                                                      scores[hemi]['test_captions'],
                                                                                      scores_mod_specific_captions[
                                                                                          'test_images'],
                                                                                      scores_mod_specific_images[
                                                                                          'test_captions'])])

        print("")

        scores["subject"] = subject
        scores["model_name"] = model_name
        scores["training_mode"] = training_mode
        scores["resolution"] = resolution
        scores["mode"] = mode
        scores["alpha"] = alpha

        all_scores.append(scores)

    for scores in tqdm(all_scores):
        metrics = ["overall", "test_captions", "test_images", "min(captions,images)", 'mean(imgs_agno, captions_agno)-mean(imgs_specific, captions_specific)', 'imgs_agno - imgs_specific', 'captions_agno - captions_specific']
        fig, axes = plt.subplots(nrows=len(metrics), ncols=2*len(VIEWS), subplot_kw={'projection': '3d'}, figsize=(5*len(VIEWS), len(metrics)*2))
        fsaverage = datasets.fetch_surf_fsaverage(mesh=scores['resolution'])

        for row_axes, testing_mode in zip(axes, metrics):
            cbar_max = None
            for i, view in enumerate(VIEWS):
                for j, hemi in enumerate(['left', 'right']):
                    if testing_mode in scores[hemi].keys():
                        scores_hemi = scores[hemi][testing_mode]

                        infl_mesh = fsaverage[f"infl_{hemi}"]
                        if cbar_max is None:
                            cbar_max = scores_hemi.max()
                        # print(f" | max score: {cbar_max:.2f}")
                        title = ""
                        # if hemi == "left":
                        #     title = f"{view}"
                        if row_axes[i*2+j] == row_axes[0]:
                            title = f"{testing_mode}"

                        # destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
                        # parcellation = destrieux_atlas['map_right']
                        # parcellation = destrieux_atlas[f'maps']
                        # parcellation_surf = surface.vol_to_surf(parcellation, fsaverage[f'pial_{hemi}'], interpolation="nearest", radius=5).astype(int) #TODO

                        # these are the regions we want to outline
                        # regions_dict = {'L G_pariet_inf-Angular': 'left angular gyrus'}
                        # regions_dict = {b'G_postcentral': 'Postcentral gyrus',
                        #                 b'G_precentral': 'Precentral gyrus'}

                        # get indices in atlas for these labels
                        # regions_indices = [
                        #     [i for i, l in destrieux_atlas['labels'] if l == region][0]
                        #     for region in regions_dict
                        # ]
                        # regions_indices = [
                        #     np.where(np.array(destrieux_atlas['labels']) == region)[0][0]
                        #     for region in regions_dict
                        # ]
                        # colors = ['g', 'b']
                        #
                        # labels = list(regions_dict.values())

                        plotting.plot_surf_stat_map(
                            infl_mesh,
                            scores_hemi,
                            hemi=hemi,
                            view=view,
                            bg_map=fsaverage[f"sulc_{hemi}"],
                            title=title,
                            axes=row_axes[i*2+j],
                            colorbar=True if row_axes[i * 2 + j] == row_axes[-1] else False,
                            threshold=COLORBAR_THRESHOLD_MIN if cbar_max > 0.5 else COLORBAR_DIFFERENCE_THRESHOLD_MIN,
                            vmax=COLORBAR_MAX if cbar_max > 0.5 else None,# cbar_max,
                            vmin=0.5 if cbar_max > 0.5 else None,
                            cmap="hot" if cbar_max > 0.5 else "cold_hot",
                            symmetric_cbar=True if cbar_max < 0.5 else "auto",
                        )
                        row_axes[i * 2 + j].legend(handles=[Circle((0, 0), radius=5, color='w', label=f"{hemi} {view}")], labelspacing=1, borderpad=0, loc='upper center', frameon=False)#bbox_to_anchor=(1.9, 0.8),

                        # plotting.plot_surf_contours(infl_mesh, parcellation_surf, labels=labels,
                        #                             levels=regions_indices, axes=row_axes[i*2+j],
                        #                             legend=True,
                        #                             colors=colors)
                    else:
                        row_axes[i*2+j].axis('off')

        title = f"{scores['model_name']}_{scores['subject']}"
        plt.suptitle(title, y=0.9)
        title += f"_alpha_{str(scores['alpha'])}"
        results_searchlight = os.path.join(RESULTS_DIR, "searchlight", scores['resolution'], scores['training_mode'], scores['mode'], f"{title}.png")
        os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
        plt.subplots_adjust(hspace=0, wspace=0, right=0.85, left=0)
        plt.savefig(results_searchlight, dpi=300, bbox_inches='tight')
        # plt.show()
        

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--subset", type=int, default=None)

    parser.add_argument("--models", type=str, nargs='+', default=['clip'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=VISION_MEAN_FEAT_KEY,
                        choices=VISION_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)
    parser.add_argument("--resolution", type=str, default="fsaverage")

    parser.add_argument("--l2-regularization-alpha", type=float, default=1e3)

    parser.add_argument("--radius", type=float, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
