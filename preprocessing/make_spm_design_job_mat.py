import argparse

from scipy.io import savemat
import numpy as np
from numpy.core.records import fromarrays
import os
import nibabel as nib
from glob import glob
from nipype.interfaces.base import Bunch
import pandas as pd

from data import IDS_IMAGES_TEST
from preprocessing.create_gray_matter_masks import get_gray_matter_mask_path
from utils import SUBJECTS, FMRI_BIDS_DATA_DIR, FMRI_BETAS_DIR, FMRI_PREPROCESSING_DATASINK_DIR


def get_condition_names(trial):
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


def preprocess_event_files(event_files):
    r"""
    loads all event tsv files into a single pandas dataframe.
    - fixes onset times
    - adds run regressors (if stage==1)
    """
    data = []
    onset_shift = 0

    for r_idx, event_file in enumerate(event_files):
        df = pd.read_csv(event_file, sep='\t')
        df['onset'] += onset_shift
        df['glm_conditions'] = df.apply(get_condition_names, axis=1)

        onset_shift = df['onset'].iloc[-1] + df['duration'].iloc[-1]  # new onset_shift for the next run

        data.append(df)

    return pd.concat(data, ignore_index=True)


def get_sessions(preprocessed_functional_data_dir, sessions_subsample):
    if sessions_subsample:
        sessions = [f'ses-{ses_idx}' for ses_idx in sessions_subsample]
        session_dirs = [os.path.join(preprocessed_functional_data_dir, session) for session in sessions]
    else:
        print(f"Scanning for sessions in {preprocessed_functional_data_dir}")
        session_dirs = glob(os.path.join(preprocessed_functional_data_dir, 'ses-*'))
        sessions = [path.split(os.sep)[-1] for path in session_dirs]
    print(f"Sessions: {sessions}")
    return sessions, session_dirs


def load_event_files(tsv_files, condition_proc_func, log_file=None):
    events_df = condition_proc_func(tsv_files)
    condition_names = sorted(set(np.concatenate(events_df['glm_conditions'].values)))
    if 'null' in condition_names:
        condition_names.remove('null')

    # print("Number of conditions: ", len(condition_names))
    # print("Number of train image conditions:", len([c for c in condition_names if "train_image" in c]))
    # print("Number of train caption conditions:", len([c for c in condition_names if "train_caption" in c]))
    #
    # print("Number of train conditions:", len([c for c in condition_names if "train" in c]))
    # print("Number of test conditions:", len([c for c in condition_names if "test" in c]))
    imagery_conditions = [c for c in condition_names if "imagery" in c]
    # print("Number of imagery conditions:", len(imagery_conditions))
    for c in imagery_conditions:
        conds_imagery = [c in conds for conds in events_df['glm_conditions'].values]
        # print(f'number of repeats of imagery condition {c}: {np.sum(conds_imagery)}')

    if log_file is not None:
        events_df.to_csv(log_file, sep="\t")

    ###############################
    # Design-Matrix-Friendly format
    ###############################
    onsets = {cond: [] for cond in condition_names}
    durs = {cond: [] for cond in condition_names}
    orth = {cond: 0.0 for cond in condition_names}

    events_df = events_df.reset_index()
    for index, trial in events_df.iterrows():
        conditions = trial['glm_conditions']
        for condition in conditions:
            if condition != 'null':
                onsets[condition].append(trial['onset'])
                durs[condition].append(trial['duration'])

    subject_info = Bunch(
        conditions=np.array(condition_names, dtype=object),
        onsets=np.array([np.array(onsets[k])[:, np.newaxis] for k in condition_names], dtype=object),
        durations=np.array([np.array(durs[k])[:, np.newaxis] for k in condition_names], dtype=object),
        orthogonalizations=np.array([orth[k] for k in condition_names], dtype=object),
        tmod=np.zeros((len(condition_names),), dtype=object),
        pmod=np.zeros((len(condition_names),), dtype=object),
        regressor_names=None,
        regressors=None)

    condition_names = flatten(events_df.glm_conditions)
    return subject_info, condition_names


N_REALIGNMENT_AXES = 6
REALIGNMENT_AXES_IDX = range(1, N_REALIGNMENT_AXES + 1)


def define_multi_regressors(realign_files):
    n_runs = len(realign_files)
    reg_names = [f'UR{i}' for i in range(1, n_runs)]  # run regressors (1 less than total number of runs)
    reg_names += [f'Realign{i}' for i in REALIGNMENT_AXES_IDX]

    run_arrays = []
    realign_arrays = [[] for _ in range(N_REALIGNMENT_AXES)]
    total_size = 0
    for ridx in range(n_runs):
        realign = np.loadtxt(realign_files[ridx])
        total_size += realign.shape[0]
        for aidx in range(N_REALIGNMENT_AXES):
            realign_arrays[aidx].append(realign[:, aidx])

    run_start = 0
    for ridx in range(n_runs - 1):
        arr = np.zeros((total_size, 1), dtype=np.double)
        arr[run_start:run_start + realign_arrays[0][ridx].shape[0], 0] = 1.0
        run_start += realign_arrays[0][ridx].shape[0]
        run_arrays.append(arr)

    for aidx in range(N_REALIGNMENT_AXES):
        realign_arrays[aidx] = np.concatenate(realign_arrays[aidx])[:, np.newaxis]

    reg_arrays = np.concatenate((run_arrays, realign_arrays))

    # fill an empty np array of type object, otherwise the size check for the rec array doesn't pass
    x = np.empty(len(reg_arrays), dtype=object)
    for i in range(len(reg_arrays)):
        x[i] = reg_arrays[i]

    return fromarrays([reg_names, x], names=['name', 'val'])


def event_file_path(raw_fmri_data_dir, session, subject, task_name, run):
    raw_fmri_subj_data_dir = str(os.path.join(raw_fmri_data_dir, subject))
    return os.path.join(
        raw_fmri_subj_data_dir, session, "func",
        f"{subject}_{session}_task-{task_name}_{run}_events.tsv"
    )


def process_scans(subject, task_name, args, event_file_path_func):
    preprocessed_functional_data_dir = os.path.join(args.preprocessing_datasink_dir, "coregistered", subject)
    realignment_data_dir = os.path.join(args.preprocessing_datasink_dir, "realignment")

    scans = []
    event_files = []
    realign_files = []
    sessions, session_dirs = get_sessions(preprocessed_functional_data_dir, args.sessions)
    for session, session_dir in zip(sessions, session_dirs):
        print(f"Scanning for runs in {session_dir}")
        run_ids = [fname.split(f"{task_name}_")[1].split('_bold.nii')[0] for fname in
                   glob(os.path.join(session_dir, f'rrasub*{task_name}_run-*_bold.nii'))]
        print(f"Runs: {run_ids}")
        for run in run_ids:
            event_file = event_file_path_func(args.raw_data_dir, session, subject, task_name, run)
            event_files.append(event_file)
            realign_file = os.path.join(
                realignment_data_dir, subject, session,
                f'rp_a{subject}_{session}_task-{task_name}_{run}_bold.txt'
            )
            realign_files.append(realign_file)
            run_file = os.path.join(
                session_dir,
                f'rra{subject}_{session}_task-{task_name}_{run}_bold.nii'
            )
            run_nii = nib.load(run_file)
            run_size = run_nii.shape[-1]
            for s in range(1, run_size + 1):
                scans.append(f"{run_file},{s}")

    return scans, event_files, realign_files


def get_base_fmri_spec(units, RT, fmri_t, fmri_t0, derivs, VOLT, GLOBAL, mthresh, mask, CVI):
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


def define_fmri_betas_jobs(output_dir, subject, task_name, args, event_file_path_func, condition_proc_func):
    #####################
    # fmri parameters:
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
    mask = get_gray_matter_mask_path(subject)

    # serial correlation (don't change)
    CVI = 'AR(1)'

    fmri_spec = get_base_fmri_spec(units, RT, fmri_t, fmri_t0, derivs, VOLT, GLOBAL, mthresh, mask, CVI)

    fmri_spec['dir'] = np.array([output_dir], dtype=object)

    scans, event_files, realign_files = process_scans(subject, task_name, args, event_file_path_func)

    fmri_spec['sess']['scans'] = np.array(scans, dtype=object)[:, np.newaxis]
    fmri_spec['sess']['regress'] = define_multi_regressors(realign_files)

    # conditions
    conditions, condition_names = load_event_files(
        event_files,
        condition_proc_func=condition_proc_func,
        log_file=os.path.join(output_dir, 'dmlog_stage_1.tsv')
    )

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

    return jobs, condition_names


def flatten(xss):
    return [x for xs in xss for x in xs]


def run(args):
    for subject in args.subjects:
        print(subject)

        output_dir = str(os.path.join(args.output_dir, subject, "unstructured"))

        task_name = "coco"

        os.makedirs(output_dir, exist_ok=True)

        jobs, condition_names = define_fmri_betas_jobs(
            output_dir, subject, task_name, args, event_file_path_func=event_file_path,
            condition_proc_func=preprocess_event_files
        )

        print("Number of conditions: ", len(condition_names))
        print("Number of train image conditions:", len([c for c in condition_names if "train_image" in c]))
        print("Number of train caption conditions:", len([c for c in condition_names if "train_caption" in c]))

        print("Number of train conditions:", len([c for c in condition_names if "train" in c]))
        print("Number of test conditions:", len([c for c in condition_names if "test" in c]))

        savemat(os.path.join(output_dir, 'spm_job.mat'), jobs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)
    parser.add_argument("--sessions", type=str, nargs='+', default=None, help="Default value of None uses all sessions")

    parser.add_argument("--raw-data-dir", type=str, default=FMRI_BIDS_DATA_DIR)
    parser.add_argument("--preprocessing-datasink-dir", type=str, default=FMRI_PREPROCESSING_DATASINK_DIR)

    parser.add_argument("--output-dir", type=str, default=FMRI_BETAS_DIR)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
