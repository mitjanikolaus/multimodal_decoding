import os
from glob import glob

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn.plotting import plot_img
from tqdm import tqdm

from preprocessing.create_gray_matter_masks import get_gray_matter_mask_path
from preprocessing.make_spm_design_job_mat import get_sessions
from utils import SUBJECTS, FMRI_PREPROCESSING_DATASINK_DIR, RESULTS_DIR, FMRI_PREPROCESSED_DATA_DIR

REALIGNMENT_PARAMS = ["x_translation", "y_translation", "z_translation", "pitch", "roll", "yaw"]


def plot_head_motion_metrics():
    # realign_all_data = []
    # for subject in SUBJECTS:
    #     preprocessed_realignment_data_dir = os.path.join(FMRI_PREPROCESSING_DATASINK_DIR, "realignment", subject)
    #
    #     realign_subj_data = []
    #
    #     sessions, session_dirs = get_sessions(preprocessed_realignment_data_dir)
    #     for session, session_dir in zip(sessions, session_dirs):
    #         n_runs = len(glob(os.path.join(session_dir, 'rp_asub*run*_bold.txt')))
    #         runs = [f'run-{id:02d}' for id in range(1, n_runs + 1)]
    #         print(f"Runs: {runs}")
    #
    #         for run in runs:
    #             realign_file = os.path.join(
    #                 FMRI_PREPROCESSING_DATASINK_DIR, 'realignment', subject, session,
    #                 f'rp_a{subject}_{session}_task-coco_{run}_bold.txt'
    #             )
    #             realign_data = pd.read_csv(realign_file, delimiter='  ', names=REALIGNMENT_PARAMS).reset_index(names="volume")
    #             # multiply rotation parameters (radian units) by 50 in order to allow interpretation in terms of millimeters of displacement for a circle of diameter 100 mm (Power et al., 2012)
    #             realign_data['pitch'] = realign_data['pitch'] * 50
    #             realign_data['roll'] = realign_data['roll'] * 50
    #             realign_data['yaw'] = realign_data['yaw'] * 50
    #
    #             for param in REALIGNMENT_PARAMS:
    #                 realign_data[f'{param}_shifted'] = realign_data[param].shift(1, fill_value=0)
    #                 realign_data.loc[0, f'{param}_shifted'] = 0
    #
    #             # Framewise displacement (FD), calculated as the sum of the absolute differences of the motion parameters for successive pairs of volumes
    #             fd = []
    #             for i, row in realign_data.iterrows():
    #                 fd.append(np.sum([np.abs(row[col_name] - row[f"{col_name}_shifted"]) for col_name in REALIGNMENT_PARAMS]))
    #             realign_data['framewise_displacement'] = fd
    #
    #             for param in REALIGNMENT_PARAMS:
    #                 del realign_data[f'{param}_shifted']
    #
    #             realign_data = realign_data.melt(id_vars = ['volume'])
    #
    #             realign_data['subject'] = subject
    #             realign_data['session'] = session.replace('ses-', '')
    #             realign_data['run'] = run.replace('run-', '')
    #             # print(realign_data)
    #
    #             # sns.lineplot(data=realign_data, y='value', hue='variable', x='volume')
    #             # plt.ylim(-2, 2)
    #             # plt.show()
    #             realign_subj_data.append(realign_data)
    #
    #     realign_subj_data = pd.concat(realign_subj_data, ignore_index=True)
    #
    #     realign_all_data.append(realign_subj_data)
    #
    #     # sns.lineplot(data=realign_subj_data, y='value', hue='variable', x='session')
    #     # plt.show()
    #
    # realign_all_data = pd.concat(realign_all_data, ignore_index=True)
    # realign_all_data.to_csv('realign_data.csv')

    realign_all_data = pd.read_csv('realign_data.csv', index_col=0)

    # fd_data = realign_all_data.loc[(slice(None), ['framewise_displacement'], slice(None))]
    # sns.lineplot(data=fd_data, y='distance', hue='subject', x='session')
    # plt.ylabel('Framewise displacement (mm)')
    # plt.xlabel('Session')
    # # plt.ylim((0, ))
    # plt.show()

    realign_all_data.reset_index(inplace=True)
    realign_all_data_metrics = realign_all_data[realign_all_data.variable != 'framewise_displacement']
    g = sns.FacetGrid(realign_all_data_metrics, col="subject", col_wrap=2, aspect=1.5)
    g.map(sns.lineplot, "session", "value", "variable", hue_order=REALIGNMENT_PARAMS)
    g.set(ylim=(-0.4, 0.4), xticks=range(1, realign_all_data_metrics.session.max()))
    g.add_legend()
    for i, axis in enumerate(g.axes):
        axis.set_title(f'subject {i + 1}')
        if i % 2 == 0:
            axis.set_ylabel('distance (mm)')
    plt.savefig(os.path.join(RESULTS_DIR, 'head_motion.png'), dpi=300)

    realign_all_data_fd = realign_all_data[realign_all_data.variable == 'framewise_displacement']
    g = sns.FacetGrid(realign_all_data_fd, col="subject", col_wrap=2, aspect=1.5)
    g.map(sns.lineplot, "session", "value")
    g.set(ylim=(0, 0.4), yticks=[0, 0.1, 0.2, 0.3, 0.4], xticks=range(1, realign_all_data_fd.session.max()))
    for i, axis in enumerate(g.axes):
        axis.set_title(f'subject {i + 1}')
        if i % 2 == 0:
            axis.set_ylabel('distance (mm)')
    plt.savefig(os.path.join(RESULTS_DIR, 'framewise_displacement.png'), dpi=300)

    # realign_all_data_avgd = realign_all_data.groupby(['subject', 'variable', 'session']).agg(distance=('value', 'mean'))

    # realign_all_data_avgd.reset_index(inplace=True)
    # realign_all_data_metrics = realign_all_data_avgd[realign_all_data_avgd.variable != 'framewise_displacement']
    # g = sns.FacetGrid(realign_all_data_metrics, col="subject", col_wrap=2, aspect=1.5)
    # g.map(sns.lineplot, "session", "distance", "variable",  hue_order=REALIGNMENT_PARAMS)
    # g.set(ylim=(-0.4, 0.4), xticks=range(1, realign_all_data_metrics.session.max()))
    # g.add_legend()
    # for i, axis in enumerate(g.axes):
    #     axis.set_title(f'subject {i+1}')
    #     if i % 2 == 0:
    #         axis.set_ylabel('distance (mm)')
    # plt.savefig(os.path.join(RESULTS_DIR, 'head_motion.png'), dpi=300)
    #
    # realign_all_data_fd = realign_all_data_avgd[realign_all_data_avgd.variable == 'framewise_displacement']
    # g = sns.FacetGrid(realign_all_data_fd, col="subject", col_wrap=2, aspect=1.5)
    # g.map(sns.lineplot, "session", "distance")
    # g.set(ylim=(0, 0.4), xticks=range(1, realign_all_data_fd.session.max()))
    # for i, axis in enumerate(g.axes):
    #     axis.set_title(f'subject {i+1}')
    #     if i % 2 == 0:
    #         axis.set_ylabel('distance (mm)')
    # plt.savefig(os.path.join(RESULTS_DIR, 'framewise_displacement.png'), dpi=300)



def run():
    plot_head_motion_metrics()


if __name__ == "__main__":
    run()
