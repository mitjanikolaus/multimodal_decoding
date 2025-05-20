import argparse

import pandas as pd
from scipy.io import savemat
import os

from data import IMAGERY, SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, \
    SPLIT_TEST_IMAGE_UNATTENDED, SPLIT_TEST_CAPTION_UNATTENDED, SPLIT_IMAGERY
from preprocessing.make_spm_design_job_mat import define_fmri_betas_jobs
from utils import ATTENTION_MOD_SUBJECTS, ATTENTION_MOD_FMRI_RAW_BIDS_DATA_DIR, \
    ATTENTION_MOD_FMRI_PREPROCESSED_DATASINK_DIR, FMRI_BETAS_DIR, ATTENTION_MOD_UNSTRUCTURED_DIR_NAME

FIXATION = "fixation"
FIXATION_WHITESCREEN = "fixation_whitescreen"
IMAGERY_INSTRUCTION = "imagery_instruction"

ID_TO_TRIAL_TYPE = {
    -1: FIXATION_WHITESCREEN,
    0: FIXATION,
    10: SPLIT_IMAGERY,
    11: IMAGERY_INSTRUCTION,
    4: SPLIT_TEST_IMAGE_ATTENDED,
    5: SPLIT_TEST_CAPTION_ATTENDED,
    40: SPLIT_TEST_IMAGE_UNATTENDED,
    50: SPLIT_TEST_CAPTION_UNATTENDED
}

TRIAL_TYPE_TO_ID = {
    trial_type: id for id, trial_type in ID_TO_TRIAL_TYPE.items()
}


def get_attention_mod_condition_names(trial):
    conditions = []
    trial_type = ID_TO_TRIAL_TYPE[trial['trial_type']]
    if trial_type in [FIXATION, FIXATION_WHITESCREEN]:
        assert trial['stim_name'] == 'Fix'
        conditions.append(trial_type)

    elif trial_type == IMAGERY_INSTRUCTION:
        assert trial['stim_name'] == 'ImgInstr'
        conditions.append(trial_type)

    elif trial_type == IMAGERY:
        stim_id = trial['condition_name']
        conditions.append(f"{trial_type}_{stim_id}")

    elif (trial['one_back'] != 0) or (trial['subj_resp'] != 0):
        if trial['one_back'] != 0:
            conditions.append('one_back')
        if trial['subj_resp'] != 0:
            conditions.append('subj_resp')
    else:
        assert trial['condition_name'] != 0
        stim_id = trial['condition_name']
        assert trial_type in [SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, SPLIT_TEST_IMAGE_UNATTENDED,
                              SPLIT_TEST_CAPTION_UNATTENDED]

        conditions.append(f"{trial_type}_{stim_id}")

    if len(conditions) == 0:
        print(f'Unknown condition for trial: {trial}')
    if (len(conditions) > 1) and not (conditions == ["one_back", "subj_resp"]):
        print(f'Multiple conditions for trial: {trial}: {conditions}')
    return conditions


def preprocess_additional_test_event_files(event_files):
    data = []
    onset_shift = 0

    for r_idx, event_file in enumerate(event_files):
        df = pd.read_csv(event_file, sep='\t')
        df['onset'] += onset_shift
        trial_types = df.trial_type.unique()
        if (TRIAL_TYPE_TO_ID[SPLIT_TEST_IMAGE_ATTENDED] in trial_types) and (
                TRIAL_TYPE_TO_ID[SPLIT_TEST_CAPTION_ATTENDED] in trial_types):
            raise RuntimeError(f"block with attention to both modalities: {trial_types}")

        df['glm_conditions'] = df.apply(get_attention_mod_condition_names, axis=1)

        onset_shift = df['onset'].iloc[-1] + df['duration'].iloc[-1]  # new onset_shift for the next run

        data.append(df)

    return pd.concat(data, ignore_index=True)


def event_file_path_additional_test(raw_fmri_subj_data_dir, session, subject, task_name, run):
    return os.path.join(
        raw_fmri_subj_data_dir, "Extra", "events",
        f"{subject}_{session}_task-{task_name}_{run}_events.tsv"
    )


def run(args):
    for subject in args.subjects:
        print(subject)

        #####################
        # fmri parameters:
        #####################

        task_name = "FUNCIMGRun1Back167dynSENSE12"

        output_dir = str(os.path.join(args.output_dir, subject, ATTENTION_MOD_UNSTRUCTURED_DIR_NAME))
        os.makedirs(output_dir, exist_ok=True)

        jobs, conditions = define_fmri_betas_jobs(
            output_dir, subject, task_name, args, event_file_path_func=event_file_path_additional_test,
            condition_proc_func=preprocess_additional_test_event_files
        )
        print("Number of conditions: ", len(conditions))

        for cond in [
            SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, SPLIT_TEST_IMAGE_UNATTENDED,
            SPLIT_TEST_CAPTION_UNATTENDED, IMAGERY_INSTRUCTION
        ]:
            print(f"Number of {cond} conditions: {len([c for c in conditions if cond in c])}")

        print(
            f"Number of imagery conditions: {len([c for c in conditions if (IMAGERY in c) and not (IMAGERY_INSTRUCTION in c)])}")

        print("")
        unique_conds = set(conditions)
        for cond in [SPLIT_TEST_IMAGE_ATTENDED, SPLIT_TEST_CAPTION_ATTENDED, SPLIT_TEST_IMAGE_UNATTENDED,
                     SPLIT_TEST_CAPTION_UNATTENDED]:
            print(f"Number of unique {cond} conditions: {len([c for c in unique_conds if cond in c])}")
        print(
            f"Number of unique imagery conditions: {len([c for c in unique_conds if (IMAGERY in c) and not (IMAGERY_INSTRUCTION in c)])}")

        savemat(os.path.join(output_dir, 'spm_job.mat'), jobs)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=ATTENTION_MOD_SUBJECTS)
    parser.add_argument("--sessions", type=str, nargs='+', default=None, help="Default value of None uses all sessions")

    parser.add_argument("--raw-data-dir", type=str, default=ATTENTION_MOD_FMRI_RAW_BIDS_DATA_DIR)
    parser.add_argument("--preprocessing-datasink-dir", type=str, default=ATTENTION_MOD_FMRI_PREPROCESSED_DATASINK_DIR)

    parser.add_argument("--output-dir", type=str, default=FMRI_BETAS_DIR)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
