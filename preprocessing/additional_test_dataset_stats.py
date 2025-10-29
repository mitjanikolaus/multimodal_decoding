import os
from collections import Counter
from glob import glob

import pandas as pd

from data import IMAGERY
from preprocessing.make_spm_design_job_mat_additional_test import ID_TO_TRIAL_TYPE, TEST_IMAGES_ATTENDED, \
    TEST_CAPTIONS_ATTENDED, TEST_IMAGES_UNATTENDED, TEST_CAPTIONS_UNATTENDED
from utils import SUBJECTS_ADDITIONAL_TEST, ADDITIONAL_TEST_FMRI_RAW_BIDS_DATA_DIR

if __name__ == "__main__":
    all_trials = []
    for subject in SUBJECTS_ADDITIONAL_TEST:
        print(subject)
        path = os.path.join(ADDITIONAL_TEST_FMRI_RAW_BIDS_DATA_DIR, subject)
        print(f"Scanning for sessions in {path}")
        session_dirs = glob(os.path.join(path, 'ses-*'))
        print(f'found {len(session_dirs)} sessions')
        rep_counter = Counter()
        for session_dir in session_dirs:
            func_scans_dir = os.path.join(session_dir, 'func')
            event_files = glob(os.path.join(func_scans_dir, '*.tsv'))
            print(f'found {len(event_files)} event files')
            for event_file in event_files:
                events = pd.read_csv(event_file, delimiter='\t')
                trial_types = events.trial_type.apply(lambda x: ID_TO_TRIAL_TYPE[x])
                conds = [TEST_IMAGES_ATTENDED, TEST_CAPTIONS_ATTENDED, TEST_IMAGES_UNATTENDED, TEST_CAPTIONS_UNATTENDED, IMAGERY]
                trials = [e + "-" + t for e, t in zip(events.condition_name.astype(str), trial_types) if t in conds]
                print(f'num perception trials: {len([t for t in trial_types if t in [TEST_IMAGES_ATTENDED, TEST_CAPTIONS_ATTENDED, TEST_IMAGES_UNATTENDED, TEST_CAPTIONS_UNATTENDED]])}')
                print(f'num imagery trials: {len([t for t in trial_types if t in [IMAGERY]])}')
                rep_counter.update(trials)
        entry = {'subject': subject}

        stimuli_counter = Counter()
        for t, c in rep_counter.items():
            stimuli_counter.update([t.split('-')[1]]) # use only
        entry.update({t: c for t, c in stimuli_counter.items()})
        all_trials.append(entry)

    df = pd.DataFrame.from_records(all_trials, index='subject')
    df = df[[TEST_IMAGES_ATTENDED, TEST_CAPTIONS_ATTENDED, TEST_IMAGES_UNATTENDED, TEST_CAPTIONS_UNATTENDED, IMAGERY]]
    print(df)
    print(df.to_latex())