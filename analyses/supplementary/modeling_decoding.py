import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_CLASSES = 70
N_TRAIN_SAMPLES_PER_CLASS = 100
N_TEST_SAMPLES_PER_CLASS = 1000

N_VOXELS_FMRI = 200

STDDEV_WITHIN_CLASS = 1.5


def generate_dummy_fmri_data(n_train_samples_per_class, seed, second_modality=None):
    np.random.seed(seed)
    data_classes = np.random.uniform(size=(N_CLASSES, N_VOXELS_FMRI))
    if second_modality is not None and second_modality in ["independent", "half_independent_half_same"]:
        data_classes_mod_2 = np.random.uniform(size=(N_CLASSES, N_VOXELS_FMRI))

    train_data_mod1 = []
    train_labels_mod1 = []
    train_data_mod2 = []
    train_labels_mod2 = []
    test_data_mod1 = []
    test_labels_mod1 = []
    test_data_mod2 = []
    test_labels_mod2 = []

    for c, class_proto in enumerate(data_classes):
        class_train_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                          size=(n_train_samples_per_class, N_VOXELS_FMRI))
        train_data_mod1.append(class_train_data)
        train_labels_mod1.extend([c] * len(class_train_data))
        mod_2_class_train_data = None
        mod_2_class_test_data = None
        if second_modality is not None:
            if second_modality == "gauss_same_stddev":
                mod_2_class_train_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))

                mod_2_class_test_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                       size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))

            elif second_modality == "gauss_smaller_stddev":
                mod_2_class_train_data = class_proto + np.random.normal(scale=0.5 * STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                mod_2_class_test_data = class_proto + np.random.normal(scale=0.5 * STDDEV_WITHIN_CLASS,
                                                                       size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))

            elif second_modality == "gauss_higher_stddev":
                mod_2_class_train_data = class_proto + np.random.normal(scale=2 * STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                mod_2_class_test_data = class_proto + np.random.normal(scale=2 * STDDEV_WITHIN_CLASS,
                                                                       size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))

            elif second_modality == "offset":
                mod_2_class_train_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                mod_2_class_train_data += 1

                mod_2_class_test_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                       size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
                mod_2_class_test_data += 1
            elif second_modality == "inverse":
                mod_2_class_train_data = -1 * class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                             size=(n_train_samples_per_class,
                                                                                   N_VOXELS_FMRI))

                mod_2_class_test_data = -1 * class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                            size=(
                                                                            N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))

            elif second_modality == "orthogonal":
                def get_orthogonal(k):
                    x = np.random.randn(k.shape[0])
                    x -= x.dot(k) * k
                    x /= np.linalg.norm(x)
                    return x

                mod2_class_proto = get_orthogonal(class_proto)
                mod_2_class_train_data = mod2_class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                             size=(n_train_samples_per_class,
                                                                                   N_VOXELS_FMRI))

                mod_2_class_test_data = mod2_class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                            size=(
                                                                            N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
            elif second_modality == "independent":
                mod_2_class_train_data = data_classes_mod_2[c] + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                  size=(n_train_samples_per_class,
                                                                                        N_VOXELS_FMRI))

                mod_2_class_test_data = data_classes_mod_2[c] + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                 size=(N_TEST_SAMPLES_PER_CLASS,
                                                                                       N_VOXELS_FMRI))
            elif second_modality == "half_independent_half_same":
                half_size = round(len(data_classes_mod_2[c]) / 2)
                mod_2_class_train_data_proto = np.concatenate(
                    (data_classes_mod_2[c][:half_size], class_proto[half_size:]))

                mod_2_class_train_data = mod_2_class_train_data_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                         size=(
                                                                                         n_train_samples_per_class,
                                                                                         N_VOXELS_FMRI))

                mod_2_class_test_data = mod_2_class_train_data_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                        size=(N_TEST_SAMPLES_PER_CLASS,
                                                                                              N_VOXELS_FMRI))

            elif second_modality == "three_quarters_noise_quarter_same":
                quarter_size = round(len(class_proto) / 4)
                mod_2_class_train_data_proto = np.concatenate((np.repeat(0, quarter_size*3), class_proto[:quarter_size]))

                mod_2_class_train_data = mod_2_class_train_data_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                         size=(
                                                                                         n_train_samples_per_class,
                                                                                         N_VOXELS_FMRI))

                mod_2_class_test_data = mod_2_class_train_data_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                        size=(N_TEST_SAMPLES_PER_CLASS,
                                                                                              N_VOXELS_FMRI))

            elif second_modality == "half_noise_half_same":
                half_size = round(len(class_proto) / 2)
                mod_2_class_train_data_proto = np.concatenate((np.repeat(0, half_size), class_proto[half_size:]))

                mod_2_class_train_data = mod_2_class_train_data_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                         size=(
                                                                                         n_train_samples_per_class,
                                                                                         N_VOXELS_FMRI))

                mod_2_class_test_data = mod_2_class_train_data_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                        size=(N_TEST_SAMPLES_PER_CLASS,
                                                                                              N_VOXELS_FMRI))
            elif second_modality == "quarter_noise_three_quarters_same":
                quarter_size = round(len(class_proto) / 4)
                mod_2_class_train_data_proto = np.concatenate((np.repeat(0, quarter_size), class_proto[quarter_size:]))

                mod_2_class_train_data = mod_2_class_train_data_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                         size=(
                                                                                         n_train_samples_per_class,
                                                                                         N_VOXELS_FMRI))

                mod_2_class_test_data = mod_2_class_train_data_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                                        size=(N_TEST_SAMPLES_PER_CLASS,
                                                                                              N_VOXELS_FMRI))
            elif second_modality == "just_noise":
                mod_2_class_train_data = np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                          size=(n_train_samples_per_class,
                                                                N_VOXELS_FMRI))
                mod_2_class_test_data = np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                         size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
            else:
                raise RuntimeError("Unknown second modality option: ", second_modality)

        if mod_2_class_train_data is not None:
            train_data_mod2.append(mod_2_class_train_data)
            train_labels_mod2.extend([c] * len(mod_2_class_train_data))

        if mod_2_class_test_data is not None:
            test_data_mod2.append(mod_2_class_test_data)
            test_labels_mod2.extend([c] * len(mod_2_class_test_data))

        class_test_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                         size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
        test_data_mod1.append(class_test_data)
        test_labels_mod1.extend([c] * len(class_test_data))

    train_data_mod1 = np.concatenate(train_data_mod1)

    test_data_mod1 = np.concatenate(test_data_mod1)
    if len(train_data_mod2) > 0:
        train_data_mod2 = np.concatenate(train_data_mod2)
        test_data_mod2 = np.concatenate(test_data_mod2)

    return train_data_mod1, train_labels_mod1, train_data_mod2, train_labels_mod2, test_data_mod1, test_labels_mod1, test_data_mod2, test_labels_mod2


def train_and_eval(args, n_train_samples_per_class, condition=None, second_modality=None):
    results_condition = []
    for decoder_type in ["modality_agnostic", "modality_specific_mod1", "modality_specific_mod2"]:

        test_scores_mod_1 = []
        test_scores_mod_2 = []
        for seed in range(10):
            train_data_mod1, train_labels_mod1, train_data_mod2, train_labels_mod2, test_data_mod1, test_labels_mod1, test_data_mod2, test_labels_mod2 = generate_dummy_fmri_data(
                n_train_samples_per_class,
                seed=seed,
                second_modality=second_modality)
            clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=args.l2_regularization_alpha))
            if decoder_type == "modality_agnostic":
                train_data = np.concatenate((train_data_mod1, train_data_mod2))
                train_labels = np.concatenate((train_labels_mod1, train_labels_mod2))
                clf.fit(train_data, train_labels)
            elif decoder_type == "modality_specific_mod1":
                clf.fit(train_data_mod1, train_labels_mod1)
            elif decoder_type == "modality_specific_mod2":
                clf.fit(train_data_mod2, train_labels_mod2)

            test_acc_mod1 = np.mean(clf.predict(test_data_mod1) == test_labels_mod1)
            test_scores_mod_1.append(test_acc_mod1)
            if len(test_data_mod2) > 0:
                test_acc_mod2 = np.mean(clf.predict(test_data_mod2) == test_labels_mod2)
                test_scores_mod_2.append(test_acc_mod2)

        print(f"MEAN modality 1: {np.mean(test_scores_mod_1):.2f}")
        print(f"MEAN modality 2: {np.mean(test_scores_mod_2):.2f}\n")

        for score_mod_1, score_mod_2 in zip(test_scores_mod_1, test_scores_mod_2):
            results_condition.append({"condition": condition, "acc": score_mod_1, "modality": "mod1",
                                      "decoder_type": decoder_type})
            results_condition.append({"condition": condition, "acc": score_mod_2, "modality": "mod2",
                                      "decoder_type": decoder_type})
            results_condition.append({"condition": condition, "acc": np.mean((score_mod_1, score_mod_2)), "modality": "avg",
                                      "decoder_type": decoder_type})

    return results_condition


def run(args):
    results = []

    # condition = "BASELINE\nM1=x+GAUSS(0, sigma_1)"
    # print(condition)
    # # scores_baseline_mod_1, scores_mod_2 = train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS)
    # # add_to_results(scores_baseline_mod_1, scores_mod_2, results, condition)
    # # scores_baseline_mod_1, scores_mod_2 = train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, modality_agnostic=True)
    # # add_to_results(scores_baseline_mod_1, scores_mod_2, results, condition, modality_agnostic=True)
    # results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition))

    condition = "SAME STDDEV\nM2=x+GAUSS(0, sigma_1)"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "gauss_same_stddev"))

    condition = "SMALLER STDDEV\nM2=x+GAUSS(0, 0.5*sigma_1)"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "gauss_smaller_stddev"))

    condition = "HIGHER STDDEV\nM2=x+GAUSS(0, 2*sigma_1)"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "gauss_higher_stddev"))

    condition = "OFFSET\nM2=x+GAUSS(1, sigma_1)"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "offset"))

    condition = "INVERSE\nM2=-1*x+GAUSS(0, sigma_1)"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "inverse"))

    condition = "JUST NOISE\nM2=GAUSS(0, sigma_1)"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "just_noise"))

    condition = "INDEPENDENT\nM2=y+GAUSS(0, sigma_1)"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "independent"))

    condition = "HALF INDEPENDENT,\nHALF SAME"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "half_independent_half_same"))

    # condition = "THREE QUARTERS NOISE,\nQUARTER SAME"
    # print(condition)
    # results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "three_quarters_noise_quarter_same"))

    condition = "HALF NOISE,\nHALF SAME"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "half_noise_half_same"))

    condition = "QUARTER NOISE,\nTHREE QUARTERS SAME"
    print(condition)
    results.extend(train_and_eval(args, N_TRAIN_SAMPLES_PER_CLASS, condition, "quarter_noise_three_quarters_same"))

    results = pd.DataFrame.from_records(results)
    sns.catplot(data=results, kind="bar", y="condition", x="acc", row="modality", hue="decoder_type", aspect=4)
    # sns.barplot(data=results, y="condition", x="acc", hue="modality")
    # plt.axhline(x=np.mean(scores_baseline_mod_1), color='black', linestyle="--", linewidth=0.7)
    # plt.xticks(rotation=85)
    plt.tight_layout()
    plt.ylabel("")
    plt.savefig("results/decoding_simulation/accs.png", dpi=300)

    results_diff = []
    diff_accs = results[(results.modality == "mod1") & (results.decoder_type == "modality_agnostic")].acc.values - results[
        (results.modality == "mod1") & (results.decoder_type == "modality_specific_mod1")].acc.values
    diff_df = results[(results.modality == "mod1") & (results.decoder_type == "modality_agnostic")].copy()
    diff_df['acc'] = diff_accs
    diff_df['metric'] = 'mod1_agnostic - mod1_specific'
    diff_df['decoder_type'] = None
    results_diff.append(diff_df)

    diff_accs = results[(results.modality == "mod2") & (results.decoder_type == "modality_agnostic")].acc.values - results[
        (results.modality == "mod2") & (results.decoder_type == "modality_specific_mod2")].acc.values
    diff_df = results[(results.modality == "mod2") & (results.decoder_type == "modality_agnostic")].copy()
    diff_df['acc'] = diff_accs
    diff_df['metric'] = 'mod2_agnostic - mod2_specific'
    diff_df['decoder_type'] = None
    results_diff.append(diff_df)

    results_diff_df = pd.concat(results_diff, ignore_index=True)

    min_acc = np.min((results_diff_df[results_diff_df.metric == "mod1_agnostic - mod1_specific"].acc.values, results_diff_df[results_diff_df.metric == "mod2_agnostic - mod2_specific"].acc.values), axis=0)
    diff_df = results_diff_df[results_diff_df.metric == "mod1_agnostic - mod1_specific"].copy()
    diff_df['acc'] = min_acc
    diff_df['metric'] = 'min(mod1_agnostic - mod1_specific, mod2_agnostic - mod2_specific)'
    diff_df['decoder_type'] = None
    results_diff_df = pd.concat((results_diff_df, diff_df), ignore_index=True)

    sns.catplot(data=results_diff_df, kind="bar", y="condition", x="acc", row="metric", aspect=3)
    plt.tight_layout()
    plt.savefig("results/decoding_simulation/diffs.png", dpi=300)
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    os.makedirs("results/decoding_simulation/", exist_ok=True)
    args = get_args()

    run(args)
