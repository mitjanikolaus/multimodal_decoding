import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from eval import pairwise_accuracy, ACC_IMAGERY_MOD_AGNOSTIC
from utils import SUBJECTS, HEMIS

N_PERMS = 10000


def run():
    null_distr_t_vals = []
    mean_accs = []
    for perm in range(N_PERMS):
        accs = []
        for _ in SUBJECTS:
            latents = np.random.normal(0, 1, (73, 1024))
            preds = np.random.normal(0, 1, (3, 1024))
            acc = pairwise_accuracy(latents, preds)
            accs.append(acc)

        test_res = stats.ttest_1samp(accs, popmean=0.5, alternative="greater")
        if np.isnan(test_res[0]) or np.isinf(test_res[0]):
            null_distr_t_vals.append(0)
        else:
            null_distr_t_vals.append(test_res[0])
        mean_accs.append(np.mean(accs))
        # print(test_res)

    mean_accs = np.array(mean_accs)
    null_distr_t_vals = np.array(null_distr_t_vals)

    threshold_values = []
    p_vals = [0.05, 1e-2, 1e-3, 1e-4]
    for thresh in p_vals:
        if thresh == 1 / len(null_distr_t_vals):
            val = np.max(null_distr_t_vals)
        else:
            val = np.quantile(null_distr_t_vals, 1 - thresh, method='closest_observation')
        threshold_values.append(val)
        print(f"test statistic significance cutoff for p<{thresh}: {val:.2f}")
        # print(mean_accs[null_distr_t_vals > val])
        accs_thresholded = mean_accs[null_distr_t_vals > val]
        if len(accs_thresholded) > 0:
            print('min mean acc: ')
            print(np.min(accs_thresholded))

    # t_values = pickle.load(open("/home/mitja/Downloads/t_values.p", "rb"))
    # for hemi in HEMIS:
    #     print(hemi)
    #     vals = t_values[hemi][ACC_IMAGERY_MOD_AGNOSTIC]
    #     vals = vals[~np.isnan(vals)]
    #     for p_val, thresh in zip(p_vals, threshold_values):
    #         print(f'p val: {p_val} threshold: {thresh:.2f}')
    #         print(np.mean(vals > thresh))


if __name__ == "__main__":
    run()
