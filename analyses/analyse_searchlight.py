import argparse
import numpy as np
from nilearn import datasets, plotting
import matplotlib.pyplot as plt
import os
from glob import glob
import pickle


from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, TEST_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS

from utils import VISION_MEAN_FEAT_KEY, RESULTS_DIR

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")


def run(args):
    results_regex = os.path.join(SEARCHLIGHT_OUT_DIR, f'*/*/*/*/*/*/alpha_*.p')
    results_paths = np.array(sorted(glob(results_regex)))

    for path in results_paths:
        print(path)
        testing_mode = os.path.basename(path).split("_")[-1].replace(".p", "")
        resolution = os.path.dirname(path).split("/")[-2]
        hemi = os.path.dirname(path).split("/")[-1]
        subject = os.path.dirname(path).split("/")[-3]
        model_name = os.path.dirname(path).split("/")[-5]

        scores = pickle.load(open(path, 'rb'))
        fsaverage = datasets.fetch_surf_fsaverage(mesh=resolution)
        infl_mesh = fsaverage[f"infl_{hemi}"]
        fig = plt.figure()
        title = f"{model_name} accuracy map {subject}, {hemi} hemisphere, {testing_mode} {resolution}"
        plotting.plot_surf_stat_map(
            infl_mesh,
            scores,
            view="medial",
            colorbar=True,
            threshold=0.6,
            bg_map=fsaverage[f"sulc_{hemi}"],
            title=title,
            figure=fig,
        )
        results_searchlight = os.path.join(RESULTS_DIR, "searchlight", f"{title}.png")
        os.makedirs(os.path.dirname(results_searchlight), exist_ok=True)
        plt.savefig(results_searchlight, dpi=300)
        
    # plotting.show()


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
    parser.add_argument("--resolution", type=str, default="fsaverage")

    parser.add_argument("--l2-regularization-alpha", type=float, default=1e3)

    parser.add_argument("--radius", type=float, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
