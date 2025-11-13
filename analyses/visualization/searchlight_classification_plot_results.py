import argparse
import pickle

import pandas as pd
from PIL import Image
from nilearn import datasets, plotting

import matplotlib.pyplot as plt
import os

from analyses.decoding.searchlight.searchlight import searchlight_mode_from_args
from analyses.decoding.searchlight.searchlight_classification_permutation_testing import permutation_results_dir
from analyses.visualization.plotting_utils import plot_surf_stat_map_custom
from data import ATTENTION_MOD_SPLITS
from utils import RESULTS_DIR, HEMIS, save_plot_and_crop_img, append_images, FMRI_BETAS_DIR, SUBJECTS_ADDITIONAL_TEST, \
    DEFAULT_RESOLUTION
from analyses.decoding.searchlight.searchlight_classification import get_results_file_path

CMAP_POS_ONLY = "hot"

DEFAULT_VIEWS = ["lateral", "medial", "ventral"]


def load_results(args):
    all_results = []
    for subj in args.subjects:
        for hemi in HEMIS:
            mode = searchlight_mode_from_args(args)
            results_path = get_results_file_path(hemi, subj, mode=mode)
            print('loading ', results_path)
            results = pd.read_csv(results_path, index_col=0)
            all_results.append(results)
    all_results = pd.concat(all_results, ignore_index=True)
    return all_results


def get_subject_avg_results(all_results):
    n_vertices = len(all_results.vertex.unique())
    all_results['train_test'] = all_results['train_split'] + ' -> ' + all_results['test_split']
    avg_results = all_results.groupby(['train_test', 'hemi', 'vertex']).agg({'value': 'mean'})

    assert len(avg_results) == n_vertices * len(all_results.train_test.unique()) * len(HEMIS)
    return avg_results


def plot(args, plot_acc=False):
    fsaverage = datasets.fetch_surf_fsaverage(mesh=DEFAULT_RESOLUTION)
    vals_to_plot = None
    if plot_acc:
        all_results = load_results(args)
        vals_to_plot = get_subject_avg_results(all_results)

    for training_split in ATTENTION_MOD_SPLITS:
        testing_splits = [split for split in ATTENTION_MOD_SPLITS if split != training_split]
        for testing_split in testing_splits:
            train_test = training_split + ' -> ' + testing_split
            results_path = str(os.path.join(RESULTS_DIR, "searchlight_classification", searchlight_mode_from_args(args)))
            atlas_tmp_results_dir = str(os.path.join(results_path, "tmp", f"{train_test}"))
            os.makedirs(atlas_tmp_results_dir, exist_ok=True)

            metric_name = f'{training_split}-{testing_split}'
            tfce_values_path = os.path.join(permutation_results_dir(args), f"tfce_values_{metric_name}.p")
            if not plot_acc:
                vals_to_plot = pickle.load(open(tfce_values_path, "rb"))

            result_values = dict()
            for hemi in HEMIS:
                if plot_acc:
                    result_values[hemi] = vals_to_plot.loc[train_test, hemi].value.values
                else:
                    result_values[hemi] = vals_to_plot[hemi]

            if plot_acc:
                threshold = 0.52
                cbar_min = 0.5
                cbar_max = 0.75
            else:
                threshold = 104122.75
                cbar_min = 104122.75
                cbar_max = 400000

            for hemi in HEMIS:
                for view in args.views:
                    fig = plotting.plot_surf_stat_map(
                        fsaverage[f"infl_{hemi}"],
                        result_values[hemi],
                        bg_map=fsaverage[f"sulc_{hemi}"],
                        hemi=hemi,
                        view=view,
                        colorbar=False,
                        threshold=threshold,
                        vmax=cbar_max,
                        vmin=cbar_min,
                        cmap=CMAP_POS_ONLY,
                    )

                    title = f"{view}_{hemi}"
                    path = os.path.join(atlas_tmp_results_dir, f"{title}.png")
                    save_plot_and_crop_img(path)
                    print(f"saved {path}")

            # plot for cbar:
            fig = plt.figure(figsize=(7, 6))
            plot_surf_stat_map_custom(
                fsaverage[f"infl_{HEMIS[0]}"],
                result_values[HEMIS[0]],
                hemi=HEMIS[0],
                view=args.views[0],
                colorbar=True,
                threshold=threshold,
                vmax=cbar_max,
                vmin=cbar_min,
                cmap=CMAP_POS_ONLY,
                figure=fig,
                metric=train_test,
            )
            save_plot_and_crop_img(os.path.join(atlas_tmp_results_dir, "colorbar.png"), crop_cbar=True,
                                   horizontal_cbar=True)

            create_composite_images_of_all_views(train_test, results_path)


def create_composite_images_of_all_views(train_test, results_path):
    results_values_imgs_dir = str(os.path.join(results_path, "tmp", f"{train_test}"))

    images_lateral = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in
                      ["lateral"] for hemi in HEMIS]
    images_medial = [Image.open(os.path.join(results_values_imgs_dir, f"{view}_{hemi}.png")) for view in ["medial"]
                     for hemi in HEMIS]

    imgs_ventral = [Image.open(os.path.join(results_values_imgs_dir, f"ventral_{hemi}.png")) for hemi in HEMIS]
    img_ventral = append_images(images=imgs_ventral, horizontally=False)

    img_medial = append_images(images=images_medial, padding=20)

    img_lateral = append_images(images=images_lateral, padding=20)

    img_colorbar = Image.open(os.path.join(results_values_imgs_dir, "colorbar.png"))
    offset_size = (img_colorbar.size[0], int(img_lateral.size[1] - img_colorbar.size[1]))
    image_whitespace = Image.new('RGBA', offset_size, color=(255, 255, 255, 0))
    img_colorbar = append_images([image_whitespace, img_colorbar], horizontally=False)

    img_row_1 = append_images([img_lateral], padding=10)
    img_row_2 = append_images([img_medial], padding=10)
    img_row_3 = append_images([img_ventral, img_colorbar], padding=30)

    # roi_legend = Image.open(os.path.join(tfce_values_imgs_dir, f"legend.png"))

    plt.figure(figsize=(10, 0.4))
    plt.text(-0.15, 0.1, train_test, fontsize=20)
    plt.axis('off')
    plt.savefig(os.path.join(results_values_imgs_dir, f"title.png", ), transparent=True, dpi=300)
    title = Image.open(os.path.join(results_values_imgs_dir, "title.png"))

    composite_image = append_images([img_row_1, img_row_2, img_row_3], padding=5, horizontally=True)
    composite_image = append_images([title, composite_image], padding=5, horizontally=False)

    path = os.path.join(results_path, "searchlight_results", f"{train_test}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    composite_image.save(path, transparent=True)  # , facecolor="black")
    print('saved ', path)


def create_composite_images_of_metrics(args):
    imgs = []
    for training_split in ATTENTION_MOD_SPLITS:
        testing_splits = [split for split in ATTENTION_MOD_SPLITS if split != training_split]
        for testing_split in testing_splits:
            train_test = training_split + ' -> ' + testing_split
            results_path = os.path.join(RESULTS_DIR, "searchlight_classification", searchlight_mode_from_args(args),
                                        "searchlight_results",
                                        f"{train_test}.png")
            imgs.append(Image.open(results_path))
    img_all_metrics = append_images(images=imgs, padding=20, horizontally=False)
    img_all_metrics_path = os.path.join(RESULTS_DIR, "searchlight_classification", searchlight_mode_from_args(args),
                                        "searchlight_results", f"composite.png")

    # Add background color
    background = Image.new('RGBA', img_all_metrics.size, (255, 255, 255))
    with_bg = Image.alpha_composite(background, img_all_metrics)

    with_bg.save(img_all_metrics_path)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--betas-dir", type=str, default=FMRI_BETAS_DIR)
    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS_ADDITIONAL_TEST)

    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)

    parser.add_argument("--views", nargs="+", type=str, default=DEFAULT_VIEWS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # train_test = 'test_caption_attended -> test_caption_unattended'
    # results_path = str(os.path.join(RESULTS_DIR, "searchlight_classification", searchlight_mode_from_args(args)))
    # create_composite_images_of_all_views(train_test, results_path)

    plot(args)
    create_composite_images_of_metrics(args)
