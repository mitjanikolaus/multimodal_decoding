import pandas as pd
import numpy as np
from glob import glob
import os
import seaborn as sns
from matplotlib import pyplot as plt

from utils import SUBJECTS, FMRI_DATA_DIR, RESULTS_DIR


def subject_performance(subj):
    path = os.path.join(FMRI_DATA_DIR, f'fmri_data/bids/{subj}/')
    sess = sorted(list(glob(os.path.join(path, 'ses-*'))))
    print(f"Subject: {subj} Number of sessions: {len(sess)}")

    confusion = np.zeros((2, 2)).astype('int')
    total_num_events = 0
    for ses in sess:
        events = sorted(list(glob(os.path.join(ses, "func/*events*.tsv"))))
        total_num_events += len(events)
        for event in events:
            data = pd.read_csv(event, sep='\t')
            condition = np.array(data['condition_name'])
            allowed = condition != 0

            one_back = np.array(data['one_back'])[allowed]
            response = np.array(data['subj_resp'])[allowed]

            confusion[0, 0] += np.logical_and(one_back == 0, response == 0).sum()
            confusion[0, 1] += np.logical_and(one_back == 0, response != 0).sum()
            confusion[1, 0] += np.logical_and(one_back != 0, response == 0).sum()
            confusion[1, 1] += np.logical_and(one_back != 0, response != 0).sum()

    error_rate_false_positives = 100 * confusion[0, 1] / confusion[0].sum()
    error_rate_false_negatives = 100 * confusion[1, 0] / confusion[1].sum()
    print("Total number of events: ", total_num_events)
    print(f"{' ':10s} {'stim':>6s} {'oneback':>10s} {'error %':>10s}")
    print(f"{'stim':10s} {confusion[0, 0]:6d} {confusion[0, 1]:10d} {error_rate_false_positives:10.2f}")
    print(f"{'oneback':10s} {confusion[1, 0]:6d} {confusion[1, 1]:10d} {error_rate_false_negatives:10.2f}")

    return error_rate_false_positives, error_rate_false_negatives


def get_oneback_errors(subj):
    r"""
    searches all the event files for oneback events without oneback flag
    """
    path = os.path.join(FMRI_DATA_DIR, f'fmri_data/bids/{subj}/')
    sess = sorted(list(glob(os.path.join(path, 'ses-*'))))

    for ses in sess:
        events = sorted(list(glob(os.path.join(ses, "func/*events*.tsv"))))
        for event in events:
            indices = []
            data = pd.read_csv(event, sep='\t')
            condition = np.array(data['condition_name'])
            one_back = np.array(data['one_back'])
            for idx in range(condition.shape[0] - 1):
                if condition[idx] != 0 and condition[idx] == condition[idx + 1] and one_back[idx + 1] == 0:
                    indices.append(idx + 2)
            if len(indices) != 0:
                print("oneback error: ")
                print(event, indices)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    for subj in SUBJECTS:
        fp, fn = subject_performance(subj)
        # get_oneback_errors(subj)
        results.append({"subject": subj, "metric": "false_positives", "value": fp})
        results.append({"subject": subj, "metric": "false_negatives", "value": fn})

    results = pd.DataFrame.from_records(results)
    sns.barplot(data=results, x="subject", y="value", hue="metric")
    plt.ylabel("Error rate")
    plt.savefig(os.path.join(RESULTS_DIR, "event_file_analysis.png"), dpi=300)
    plt.show()

