import os
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd




DATA_DIR_EXP2 = "/backup2/mitja/SEMREPS_LOCALIZER/SEMREPS_LOCALIZER_BIDS/"
SUBJECTS_EXP2 = ['sub-01', 'sub-02', 'sub-04', 'sub-05', 'sub-07']

TRIAL_TYPE_MAPPING = {
    -1: "whitescreen imagery background fixation", #??
    0: "fixation",
    4: "image_attended",
    5: "caption_attended",
    10: "imagery",
    11: "imagery_instruction",
    40: "image_unattended",
    50: "caption_unattended"
}


if __name__ == "__main__":
    all_trials = []
    for subject in SUBJECTS_EXP2:
        print(subject)
        path = os.path.join(DATA_DIR_EXP2, subject)
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
                trials = [e + "-" + t for e, t in zip(events.condition_name.astype(str), trial_types) if t in ['image_attended', 'caption_attended', 'image_unattended', 'caption_unattended', 'imagery']]
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