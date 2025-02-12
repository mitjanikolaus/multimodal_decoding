import os
from collections import Counter
from glob import glob

import pandas as pd

from preprocessing.make_spm_design_job_mat_attention_mod import TRIAL_TYPE_MAPPING
from utils import ATTENTION_MOD_SUBJECTS, ATTENTION_MOD_FMRI_RAW_BIDS_DATA_DIR

if __name__ == "__main__":
    all_trials = []
    for subject in ATTENTION_MOD_SUBJECTS:
        print(subject)
        path = os.path.join(ATTENTION_MOD_FMRI_RAW_BIDS_DATA_DIR, subject)
        print(f"Scanning for sessions in {path}")
        session_dirs = glob(os.path.join(path, 'ses-*'))
        rep_counter = Counter()
        for session_dir in session_dirs:
            func_scans_dir = os.path.join(session_dir, 'func')
            event_files = glob(os.path.join(func_scans_dir, '*.tsv'))
            for event_file in event_files:
                events = pd.read_csv(event_file, delimiter='\t')
                # print(events.trial_type.unique())
                trial_types = events.trial_type.apply(lambda x: TRIAL_TYPE_MAPPING[x])
                conds = ['image_attended', 'caption_attended', 'image_unattended', 'caption_unattended', 'imagery']
                trials = [e + "-" + t for e, t in zip(events.condition_name.astype(str), trial_types) if t in conds]
                rep_counter.update(trials)
        entry = {'subject': subject}

        stimuli_counter = Counter()
        for t, c in rep_counter.items():
            stimuli_counter.update([t.split('-')[1]]) # use only
        entry.update({t: c for t, c in stimuli_counter.items()})
        all_trials.append(entry)

    df = pd.DataFrame.from_records(all_trials, index='subject')
    df = df[['image_attended', 'caption_attended', 'image_unattended', 'caption_unattended', 'imagery']]
    print(df)
    print(df.to_latex())