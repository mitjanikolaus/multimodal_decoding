import os
from collections import Counter
from glob import glob

import pandas as pd

from data import IMAGERY
from preprocessing.make_spm_design_job_mat_additional_test import ID_TO_TRIAL_TYPE, SPLIT_TEST_IMAGES_ATTENDED, \
    SPLIT_TEST_CAPTIONS_ATTENDED, SPLIT_TEST_IMAGES_UNATTENDED, SPLIT_TEST_CAPTIONS_UNATTENDED
from utils import SUBJECTS_ADDITIONAL_TEST, ADDITIONAL_TEST_FMRI_RAW_BIDS_DATA_DIR

if __name__ == "__main__":
    all_trials = []
    for subject in SUBJECTS_ADDITIONAL_TEST:
        print(subject)
        path = os.path.join(ADDITIONAL_TEST_FMRI_RAW_BIDS_DATA_DIR, subject)
        print(f"Scanning for sessions in {path}")
        session_dirs = glob(os.path.join(path, 'ses-*'))
        rep_counter = Counter()
        for session_dir in session_dirs:
            func_scans_dir = os.path.join(session_dir, 'func')
            event_files = glob(os.path.join(func_scans_dir, '*.tsv'))
            for event_file in event_files:
                events = pd.read_csv(event_file, delimiter='\t')
                trial_types = events.trial_type.apply(lambda x: ID_TO_TRIAL_TYPE[x])
                conds = [SPLIT_TEST_IMAGES_ATTENDED, SPLIT_TEST_CAPTIONS_ATTENDED, SPLIT_TEST_IMAGES_UNATTENDED, SPLIT_TEST_CAPTIONS_UNATTENDED, IMAGERY]
                trials = [e + "-" + t for e, t in zip(events.condition_name.astype(str), trial_types) if t in conds]
                rep_counter.update(trials)
        entry = {'subject': subject}

        stimuli_counter = Counter()
        for t, c in rep_counter.items():
            stimuli_counter.update([t.split('-')[1]]) # use only
        entry.update({t: c for t, c in stimuli_counter.items()})
        all_trials.append(entry)

    df = pd.DataFrame.from_records(all_trials, index='subject')
    df = df[[SPLIT_TEST_IMAGES_ATTENDED, SPLIT_TEST_CAPTIONS_ATTENDED, SPLIT_TEST_IMAGES_UNATTENDED, SPLIT_TEST_CAPTIONS_UNATTENDED, IMAGERY]]
    print(df)
    print(df.to_latex())