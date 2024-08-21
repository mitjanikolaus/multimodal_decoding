###########################################
# makes a mat file for spm level1design
# - first GLM for the split case
# - with or without wbf
###########################################
import argparse

from numpy.lib.function_base import copy
from scipy.io import matlab, savemat, loadmat
import numpy as np
from numpy.core.records import fromarrays
import os
import nibabel as nib
from glob import glob
import csv
import pickle
from nipype.interfaces.base import Bunch
import pandas as pd

from preprocessing.create_gray_matter_masks import get_graymatter_mask_path
from utils import SUBJECTS, FMRI_BETAS_DIR, FMRI_PREPROCESSED_MNI_DATA_DIR, FMRI_RAW_BIDS_DATA_DIR, \
    FMRI_PREPROCESSED_DATA_DIR


##############
# EVENT Loader
##############
def get_condition_names(df, glm_stage):
    r"""
    determines the condition name for each trial (each row of df)
    """
    df = df.reset_index()
    conditions = []
    if glm_stage == 1:
        for index, trial in df.iterrows():
            if trial['stim_name'] == 'Fix':
                if trial['trial_type'] == -1:
                    conditions.append('blank')
                elif trial['trial_type'] == 0:
                    conditions.append('fixation')
            elif trial['stim_name'] == 'ImgInst':
                conditions.append('imginst')
            elif trial['stim_name'] == 'Img' and trial['imagert'] == 1:
                conditions.append(f"imagery_{trial['imagery_scene']}")
            elif trial['one_back'] != 0:
                conditions.append('oneback')
            elif trial['condition_name'] != 0:
                if trial['trial_type'] == 1 and trial['train_test'] == 1:
                    conditions.append('null')
                if trial['trial_type'] == 2 and trial['train_test'] == 1:
                    conditions.append('null')
                if trial['trial_type'] == 1 and trial['train_test'] == 2:
                    conditions.append(f"test_image_{trial['condition_name']}")
                if trial['trial_type'] == 2 and trial['train_test'] == 2:
                    conditions.append(f"test_caption_{trial['condition_name']}")
            else:
                raise Exception(f'Uncondition Trial: {trial}')
    elif glm_stage == 2:
        for index, trial in df.iterrows():
            if trial['stim_name'] == 'Fix':
                if trial['trial_type'] == -1:
                    conditions.append('null')
                elif trial['trial_type'] == 0:
                    conditions.append('null')
            elif trial['stim_name'] == 'ImgInst':
                conditions.append('null')
            elif trial['stim_name'] == 'Img' and trial['imagert'] == 1:
                conditions.append('null')
            elif trial['one_back'] != 0:
                conditions.append('null')
            elif trial['condition_name'] != 0:
                if trial['trial_type'] == 1 and trial['train_test'] == 1:
                    conditions.append(f"train_image_{trial['condition_name']}")
                if trial['trial_type'] == 2 and trial['train_test'] == 1:
                    conditions.append(f"train_caption_{trial['condition_name']}")
                if trial['trial_type'] == 1 and trial['train_test'] == 2:
                    conditions.append('null')
                if trial['trial_type'] == 2 and trial['train_test'] == 2:
                    conditions.append('null')
            else:
                raise Exception(f'Uncondition Trial: {trial}')
    return conditions


def preprocess_event_files(event_files, glm_stage):
    r"""
    loads all event tsv files into a single pandas dataframe.
    - fixes onset times
    - adds run regressors (if stage==1)
    """
    data = []
    onset_shift = 0
    run_reg_names = [f'UR{i}' for i in range(1, len(event_files))]

    for r_idx, event_file in enumerate(event_files):
        df = pd.read_csv(event_file, sep='\t')
        df['onset'] += onset_shift
        df['glm_condition'] = get_condition_names(df, glm_stage)

        if glm_stage == 1:
            run_reg_data = np.zeros((df.shape[0], len(run_reg_names)))
            if r_idx < len(run_reg_names):
                run_reg_data[:, r_idx] = 1
            run_reg_df = pd.DataFrame(run_reg_data, columns=run_reg_names)
            # for reg_idx, reg_name in enumerate(run_reg_names):
            #     df[reg_name] = run_reg_data[:,reg_idx]
            df = pd.concat([df, run_reg_df], axis=1)

        onset_shift = df['onset'].iloc[-1] + df['duration'].iloc[-1]  # new onset_shift for the next run

        data.append(df)

    return pd.concat(data, ignore_index=True)


def load_event_files_stage1(tsv_files, log_file=None):
    events_df = preprocess_event_files(tsv_files, glm_stage=1)
    condition_names = sorted(list(set(events_df['glm_condition'])))
    condition_names.remove('null')

    print(condition_names)
    print("Number of conditions:", len(condition_names))

    if log_file is not None:
        events_df.to_csv(log_file, sep="\t")

    ###############################
    # Design-Matrix-Friendly format
    ###############################
    onsets = {cond: [] for cond in condition_names}
    durs = {cond: [] for cond in condition_names}
    orth = {cond: 0.0 for cond in condition_names}
    # orth['train_image'] = 0.0
    # orth['train_caption'] = 0.0

    events_df = events_df.reset_index()
    for index, trial in events_df.iterrows():
        cond = trial['glm_condition']
        if cond != 'null':
            onsets[cond].append(trial['onset'])
            durs[cond].append(trial['duration'])

    subject_info = Bunch(
        conditions=np.array(condition_names, dtype=object),
        onsets=np.array([np.array(onsets[k])[:, np.newaxis] for k in condition_names], dtype=object),
        durations=np.array([np.array(durs[k])[:, np.newaxis] for k in condition_names], dtype=object),
        orthogonalizations=np.array([orth[k] for k in condition_names], dtype=object),
        tmod=np.zeros((len(condition_names),), dtype=object),
        pmod=np.zeros((len(condition_names),), dtype=object),
        regressor_names=None,
        regressors=None)

    return subject_info


def load_event_files_stage2(tsv_files, log_files=None):
    subject_infos = []

    for tsvf_idx, tsvf in enumerate(tsv_files):
        events_df = preprocess_event_files([tsvf], glm_stage=2)
        condition_names = sorted(list(set(events_df['glm_condition'])))
        condition_names.remove('null')

        print(condition_names)
        print("Number of conditions:", len(condition_names))

        if log_files is not None:
            events_df.to_csv(log_files[tsvf_idx], sep="\t")

        ###############################
        # Design-Matrix-Friendly format
        ###############################
        onsets = {cond: [] for cond in condition_names}
        durs = {cond: [] for cond in condition_names}
        orth = {cond: 0.0 for cond in condition_names}
        # orth['train_image'] = 0.0
        # orth['train_caption'] = 0.0

        events_df = events_df.reset_index()
        for index, trial in events_df.iterrows():
            cond = trial['glm_condition']
            if cond != 'null':
                onsets[cond].append(trial['onset'])
                durs[cond].append(trial['duration'])

        temp_condition_names = ['dummy'] + condition_names[:]
        onsets['dummy'] = [0, 0]
        durs['dummy'] = [0, 0]
        temp_onsets = np.array([np.array(onsets[k])[:, np.newaxis] for k in temp_condition_names], dtype=object)
        temp_durations = np.array([np.array(durs[k])[:, np.newaxis] for k in temp_condition_names], dtype=object)
        temp_onsets = temp_onsets[1:]
        temp_durations = temp_durations[1:]

        subject_info = Bunch(
            conditions=np.array(condition_names, dtype=object),
            # onsets=np.array([np.array(onsets[k], dtype=object)[:,np.newaxis] for k in condition_names], dtype=object),
            # durations=np.array([np.array(durs[k],dtype=object)[:,np.newaxis] for k in condition_names], dtype=object),
            onsets=temp_onsets,
            durations=temp_durations,
            orthogonalizations=np.array([orth[k] for k in condition_names], dtype=object),
            tmod=np.zeros((len(condition_names),), dtype=object),
            pmod=np.zeros((len(condition_names),), dtype=object),
            regressor_names=None,
            regressors=None)

        subject_infos.append(subject_info)
    return subject_infos


def multi_regressors(realign_files):
    n_runs = len(realign_files)

    reg_names = [f'UR{i}' for i in range(1, n_runs)]  # run regressors (1 less than total number of runs)
    reg_names += [f'Realign{i}' for i in range(1, 7)]  # 6 realignment axes

    run_arrays = []
    realign_arrays = [[] for i in range(1, 7)]
    total_size = 0
    for ridx in range(n_runs):
        realign = np.loadtxt(realign_files[ridx])
        total_size += realign.shape[0]
        for aidx in range(6):
            realign_arrays[aidx].append(realign[:, aidx])

    run_start = 0
    for ridx in range(n_runs - 1):
        arr = np.zeros((total_size, 1), dtype=np.double)
        arr[run_start:run_start + realign_arrays[0][ridx].shape[0], 0] = 1.0
        run_start += realign_arrays[0][ridx].shape[0]
        run_arrays.append(arr)

    for aidx in range(6):
        realign_arrays[aidx] = np.concatenate(realign_arrays[aidx])[:, np.newaxis]

    reg_arrays = np.array([[]] + run_arrays + realign_arrays, dtype=object)
    reg_arrays = reg_arrays[1:].astype(dtype=object, copy=True)
    return fromarrays([reg_names, reg_arrays], names=['name', 'val'])


def run(args):
    subsample_sessions = args.sessions
    # subsample_sessions = ['01', '02', '03', '05', '06', '07', '09', '11']         # None to use all sessions

    base_task_name = 'two-stage'
    if subsample_sessions:
        task_name = f'{base_task_name}_{len(subsample_sessions)}sess_{"_".join(subsample_sessions)}'
    else:
        task_name = f'{base_task_name}'

    print(task_name)
    print("Stage: ", args.stage)

    for subject in args.subjects:
        print(subject)
        preprocessed_fmri_mni_space_dir = os.path.join(FMRI_PREPROCESSED_MNI_DATA_DIR, subject)
        datasink_dir = os.path.join(FMRI_PREPROCESSED_DATA_DIR, "datasink")
        raw_fmri_subj_data_dir = os.path.join(FMRI_RAW_BIDS_DATA_DIR, subject)

        save_dir = os.path.join(FMRI_BETAS_DIR, subject, "unstructured")

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

        if args.stage == 1:
            fmri_spec = get_base_fmri_spec()

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fmri_spec['dir'] = np.array([save_dir], dtype=object)

            # listing scans
            scans = []
            event_files = []
            realign_files = []
            if subsample_sessions:
                # n_sessions = len(glob(os.path.join(data_dir,'*('+ ','.join([f'ses-{s}' for s in subsample_sessions]) +')')))
                session_list = subsample_sessions
            else:
                n_sessions = len(glob(os.path.join(preprocessed_fmri_mni_space_dir, 'ses-*')))
                session_list = [f'{i:02d}' for i in range(1, n_sessions + 1)]
            for sess_idx in session_list:
                sess_dir = os.path.join(preprocessed_fmri_mni_space_dir, f'ses-{sess_idx}')
                n_runs = len(glob(os.path.join(sess_dir, 'rarasub*run*_bold.nii')))
                for run_idx in range(1, n_runs + 1):
                    run_file = os.path.join(sess_dir,
                                            f'rara{subject}_ses-{sess_idx}_task-coco_run-{run_idx:02d}_bold.nii')
                    event_files.append(os.path.join(raw_fmri_subj_data_dir, f"ses-{sess_idx}", "func",
                                                    f"{subject}_ses-{sess_idx}_task-coco_run-{run_idx:02d}_events.tsv"))
                    realign_files.append(
                        os.path.join(
                            datasink_dir, 'realignment', subject, f'ses-{sess_idx}',
                            f'rp_a{subject}_ses-{sess_idx}_task-coco_run-{run_idx:02d}_bold.txt'
                        )
                    )
                    run_nii = nib.load(run_file)
                    run_size = run_nii.shape[-1]
                    for s in range(1, run_size + 1):
                        scans.append(f"{run_file},{s}")
            fmri_spec['sess']['scans'] = np.array(scans, dtype=object)[:, np.newaxis]

            # multi regressors
            fmri_spec['sess']['regress'] = multi_regressors(realign_files)

            # conditions
            conditions = load_event_files_stage1(
                event_files,
                log_file=os.path.join(save_dir, 'dmlog_stage_1.tsv'))

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
            savemat(os.path.join(save_dir, 'spm_lvl1_job_stage_1.mat'), jobs)

        elif args.stage == 2:
            # listing scans
            scans = []
            event_files = []
            stage_2_fmri_specs = []
            stage_2_save_dirs = []

            if subsample_sessions:
                # n_sessions = len(glob(os.path.join(data_dir,'*('+ ','.join([f'ses-{s}' for s in subsample_sessions]) +')')))
                session_list = subsample_sessions
            else:
                n_sessions = len(glob(os.path.join(preprocessed_fmri_mni_space_dir, 'ses-*')))
                session_list = [f'{i:02d}' for i in range(1, n_sessions + 1)]

            res_start = 0
            for sess_idx in session_list:
                sess_dir = os.path.join(preprocessed_fmri_mni_space_dir, f'ses-{sess_idx}')
                n_runs = len(glob(os.path.join(sess_dir, 'rarasub*run*_bold.nii')))
                for run_idx in range(1, n_runs + 1):
                    run_scans = []
                    run_file = os.path.join(sess_dir,
                                            f'rara{subject}_ses-{sess_idx}_task-coco_run-{run_idx:02d}_bold.nii')
                    res_file = save_dir
                    event_files.append(os.path.join(raw_fmri_subj_data_dir, f"ses-{sess_idx}", "func",
                                                    f"{subject}_ses-{sess_idx}_task-coco_run-{run_idx:02d}_events.tsv"))
                    run_nii = nib.load(run_file)
                    run_size = run_nii.shape[-1]
                    for s in range(1, run_size + 1):
                        run_scans.append(os.path.join(res_file, f'Res_{(res_start + s):04d}.nii'))
                    res_start += run_size
                    scans.append(run_scans)

                    fmri_spec = get_base_fmri_spec()
                    save_dir_stage2 = os.path.join(save_dir, f'run_{len(scans):03d}')
                    if not os.path.exists(save_dir_stage2):
                        os.makedirs(save_dir_stage2)
                    stage_2_save_dirs.append(save_dir_stage2)
                    fmri_spec['dir'] = np.array([save_dir_stage2], dtype=object)
                    fmri_spec['sess']['scans'] = np.array(run_scans, dtype=object)[:, np.newaxis]
                    stage_2_fmri_specs.append(fmri_spec)

            # multi regressors
            # fmri_spec['sess']['regress'] = multi_regressors(realign_files)

            # conditions
            all_conditions = load_event_files_stage2(
                event_files,
                log_files=[f"{os.path.join(d, 'dmlog_stage_2.tsv')}" for d in stage_2_save_dirs])

            for spec_idx, conditions in enumerate(all_conditions):
                stage_2_fmri_specs[spec_idx]['sess']['cond'] = fromarrays(
                    [conditions.conditions, conditions.onsets, conditions.durations, conditions.tmod, conditions.pmod,
                     conditions.orthogonalizations], names=['name', 'onset', 'duration', 'tmod', 'pmod', 'orth'],
                    shape=(len(conditions.conditions),)
                )
                stage_2_fmri_specs[spec_idx]['fact'] = fromarrays([[], []], names=['name', 'levels'])

                jobs = dict()
                jobs['jobs'] = [dict()]
                jobs['jobs'][0]['spm'] = dict()
                jobs['jobs'][0]['spm']['stats'] = dict()
                jobs['jobs'][0]['spm']['stats']['fmri_spec'] = stage_2_fmri_specs[spec_idx]
                savemat(os.path.join(stage_2_save_dirs[spec_idx], 'spm_lvl1_job_stage_2.mat'), jobs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--sessions", type=str, nargs='+', default=None, help="Default value of None uses all sessions")

    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
