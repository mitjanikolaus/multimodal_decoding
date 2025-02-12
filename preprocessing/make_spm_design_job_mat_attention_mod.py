import argparse

from scipy.io import savemat
import numpy as np
from numpy.core.records import fromarrays
import os
import nibabel as nib
from glob import glob

from data import IDS_IMAGES_TEST
from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from preprocessing.make_spm_design_job_mat import define_multi_regressors, load_event_files, get_sessions
from utils import ATTENTION_MOD_FMRI_BETAS_DIR, ATTENTION_MOD_SUBJECTS, ATTENTION_MOD_FMRI_RAW_BIDS_DATA_DIR, \
    ATTENTION_MOD_FMRI_PREPROCESSED_DATA_DIR, ATTENTION_MOD_FMRI_PREPROCESSED_MNI_DATA_DIR

TRIAL_TYPE_MAPPING = {
    -1: "fixation_whitescreen",
    0: "fixation",
    4: "image_attended",
    5: "caption_attended",
    10: "imagery",
    11: "imagery_instruction",
    40: "image_unattended",
    50: "caption_unattended"
}


def get_condition_names(trial):
    # TODO
    conditions = []
    if trial['stim_name'] == 'Fix':
        if trial['trial_type'] == -1:
            conditions.append('blank')
        elif trial['trial_type'] == 0:
            conditions.append('fixation')
    elif trial['stim_name'] == 'ImgInst':
        conditions.append('imginst')
    elif trial['stim_name'] == 'Img' and trial['imagert'] == 1:
        conditions.append(f"imagery_{trial['imagery_scene']}")
    else:
        if (trial['one_back'] != 0) or (trial['subj_resp'] != 0):
            if trial['one_back'] != 0:
                conditions.append('one_back')
            if trial['subj_resp'] != 0:
                conditions.append('subj_resp')
        else:
            if trial['condition_name'] != 0:
                stim_id = trial['condition_name']
                if trial['trial_type'] == 1:
                    if int(stim_id) in IDS_IMAGES_TEST:
                        conditions.append(f"test_image_{stim_id}")
                    else:
                        conditions.append(f"train_image_{stim_id}")
                elif trial['trial_type'] == 2:
                    if int(stim_id) in IDS_IMAGES_TEST:
                        conditions.append(f"test_caption_{stim_id}")
                    else:
                        conditions.append(f"train_caption_{stim_id}")

    if len(conditions) == 0:
        print(f'Unknown condition for trial: {trial}')
    return conditions


def run(args):
    sessions_subsample = args.sessions
    # subsample_sessions = ['01', '02', '03', '05', '06', '07', '09', '11']         # None to use all sessions

    for subject in args.subjects:
        print(subject)
        preprocessed_fmri_mni_space_dir = os.path.join(args.mni_data_dir, subject)
        realignment_data_dir = os.path.join(args.preprocessed_data_dir, "datasink", "realignment")
        raw_fmri_subj_data_dir = str(os.path.join(args.raw_data_dir, subject))

        output_dir = str(os.path.join(args.output_dir, subject, "unstructured"))

        #####################
        # 1- fmri parameters:
        #####################

        # timings
        units = 'secs'  # units for design secs/scans
        RT = 2.0  # interscan interval
        fmri_t = 46.0  # microtime resolution (16). in case of slice-timing set it to number of slices
        fmri_t0 = 23.0  # microtime onset (8). in case of slice-timing, set it to the reference slice

        # no factorial design (don't change)
        # matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});

        # hrf
        derivs = [0.0, 0.0]  # HRF derivatives

        # do not model interaction (don't change)
        VOLT = 1.0

        # no global minimization (don't change)
        GLOBAL = 'None'

        # implicit mask threhsold
        mthresh = 0.8

        # explicit mask (if set, the threshold will be ignored)
        mask = get_graymatter_mask_path(subject)

        # serial correlation (don't change)
        CVI = 'AR(1)'

        #########################
        # Generating job MAT file
        #########################
        def get_base_fmri_spec():
            fmri_spec = dict()

            fmri_spec['timing'] = dict()
            fmri_spec['timing']['units'] = units
            fmri_spec['timing']['RT'] = RT
            fmri_spec['timing']['fmri_t'] = fmri_t
            fmri_spec['timing']['fmri_t0'] = fmri_t0

            # fmri_spec['fact'] = {'name':None, 'levels':None}

            fmri_spec['bases'] = dict()
            fmri_spec['bases']['hrf'] = dict()
            fmri_spec['bases']['hrf']['derivs'] = np.array(derivs, dtype=np.double)

            fmri_spec['volt'] = VOLT
            fmri_spec['global'] = GLOBAL

            fmri_spec['mthresh'] = mthresh if len(mask) == 0 else -1 * np.inf
            fmri_spec['mask'] = np.array([mask], dtype=object)
            fmri_spec['cvi'] = CVI

            fmri_spec['sess'] = dict()
            fmri_spec['sess']['hpf'] = 128.0
            return fmri_spec

        fmri_spec = get_base_fmri_spec()

        os.makedirs(output_dir, exist_ok=True)
        fmri_spec['dir'] = np.array([output_dir], dtype=object)

        scans = []
        event_files = []
        realign_files = []
        sessions, session_dirs = get_sessions(preprocessed_fmri_mni_space_dir, sessions_subsample)
        for session, session_dir in zip(sessions, session_dirs):
            print(f"Scanning for runs in {session_dir}")
            n_runs = len(glob(os.path.join(session_dir, 'rarasub*run*_bold.nii')))
            runs = [f'run-{id:02d}' for id in range(1, n_runs + 1)]
            print(f"Runs: {runs}")
            for run in runs:
                event_file = os.path.join(
                    raw_fmri_subj_data_dir, session, "func",
                    f"{subject}_{session}_task-coco_{run}_events.tsv"
                )
                event_files.append(event_file)
                realign_file = os.path.join(
                    realignment_data_dir, subject, session,
                    f'rp_a{subject}_{session}_task-coco_{run}_bold.txt'
                )
                realign_files.append(realign_file)
                run_file = os.path.join(
                    session_dir,
                    f'rara{subject}_{session}_task-coco_{run}_bold.nii'
                )
                run_nii = nib.load(run_file)
                run_size = run_nii.shape[-1]
                for s in range(1, run_size + 1):
                    scans.append(f"{run_file},{s}")

        fmri_spec['sess']['scans'] = np.array(scans, dtype=object)[:, np.newaxis]

        # multi regressors
        fmri_spec['sess']['regress'] = define_multi_regressors(realign_files)

        # conditions
        conditions = load_event_files(
            event_files,
            log_file=os.path.join(output_dir, 'dmlog_stage_1.tsv'))

        fmri_spec['sess']['cond'] = fromarrays(
            [conditions.conditions, conditions.onsets, conditions.durations, conditions.tmod, conditions.pmod,
             conditions.orthogonalizations], names=['name', 'onset', 'duration', 'tmod', 'pmod', 'orth']
        )

        fmri_spec['fact'] = fromarrays([[], []], names=['name', 'levels'])

        jobs = dict()
        jobs['jobs'] = [dict()]
        jobs['jobs'][0]['spm'] = dict()
        jobs['jobs'][0]['spm']['stats'] = dict()
        jobs['jobs'][0]['spm']['stats']['fmri_spec'] = fmri_spec
        savemat(os.path.join(output_dir, 'spm_job.mat'), jobs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=ATTENTION_MOD_SUBJECTS)
    parser.add_argument("--sessions", type=str, nargs='+', default=None, help="Default value of None uses all sessions")

    parser.add_argument("--raw-data-dir", type=str, default=ATTENTION_MOD_FMRI_RAW_BIDS_DATA_DIR)
    parser.add_argument("--preprocessed-data-dir", type=str, default=ATTENTION_MOD_FMRI_PREPROCESSED_DATA_DIR)
    parser.add_argument("--mni-data-dir", type=str, default=ATTENTION_MOD_FMRI_PREPROCESSED_MNI_DATA_DIR)

    parser.add_argument("--output-dir", type=str, default=ATTENTION_MOD_FMRI_BETAS_DIR)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
