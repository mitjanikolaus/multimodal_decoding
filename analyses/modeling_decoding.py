import argparse

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

STDDEV_WITHIN_CLASS = 3


def generate_dummy_fmri_data(n_train_samples_per_class, seed, second_modality=None):
    np.random.seed(seed)
    data_classes = np.random.uniform(size=(N_CLASSES, N_VOXELS_FMRI))
    if second_modality is not None and second_modality in ["independent", "half_independent_half_same"]:
        data_classes_mod_2 = np.random.uniform(size=(N_CLASSES, N_VOXELS_FMRI))

    train_data = []
    train_labels = []
    test_data_mod1 = []
    test_labels_mod1 = []
    test_data_mod2 = []
    test_labels_mod2 = []

    for c, class_proto in enumerate(data_classes):
        class_train_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                          size=(n_train_samples_per_class, N_VOXELS_FMRI))
        train_data.append(class_train_data)
        train_labels.extend([c] * len(class_train_data))
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
                mod_2_class_train_data = class_proto + np.random.normal(scale=0.5*STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                mod_2_class_test_data = class_proto + np.random.normal(scale=0.5*STDDEV_WITHIN_CLASS,
                                                                 size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))

            elif second_modality == "gauss_higher_stddev":
                mod_2_class_train_data = class_proto + np.random.normal(scale=2*STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                mod_2_class_test_data = class_proto + np.random.normal(scale=2*STDDEV_WITHIN_CLASS,
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
                                                                 size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))

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
                                                                 size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
            elif second_modality == "independent":
                mod_2_class_train_data = data_classes_mod_2[c] + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))

                mod_2_class_test_data = data_classes_mod_2[c] + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                 size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
            elif second_modality == "half_independent_half_same":
                half_size = round(len(data_classes_mod_2[c])/2)
                mod_2_class_train_data = np.concatenate((data_classes_mod_2[c][:half_size], class_proto[half_size:]))

                mod_2_class_train_data = mod_2_class_train_data + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                        size=(n_train_samples_per_class,
                                                              N_VOXELS_FMRI))

                mod_2_class_test_data = mod_2_class_train_data + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                 size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
            elif second_modality == "half_noise_half_same":
                half_size = round(len(class_proto)/2)
                mod_2_class_train_data = np.concatenate((np.repeat(0, half_size), class_proto[half_size:]))

                mod_2_class_train_data = mod_2_class_train_data + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                        size=(n_train_samples_per_class,
                                                              N_VOXELS_FMRI))

                mod_2_class_test_data = mod_2_class_train_data + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                 size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
            elif second_modality == "quarter_noise_three_quarters_same":
                quarter_size = round(len(class_proto)/4)
                mod_2_class_train_data = np.concatenate((np.repeat(0, quarter_size), class_proto[quarter_size:]))

                mod_2_class_train_data = mod_2_class_train_data + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                        size=(n_train_samples_per_class,
                                                              N_VOXELS_FMRI))

                mod_2_class_test_data = mod_2_class_train_data + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                 size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
            elif second_modality == "just_noise":
                mod_2_class_train_data = np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                            size=(n_train_samples_per_class,
                                                                  N_VOXELS_FMRI))
                mod_2_class_test_data = np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                 size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
            else:
                raise RuntimeError("Unknown second modality option: ", second_modality)

        if mod_2_class_train_data is not None:
            train_data.append(mod_2_class_train_data)
            train_labels.extend([c] * len(mod_2_class_train_data))

        if mod_2_class_test_data is not None:
            test_data_mod2.append(mod_2_class_test_data)
            test_labels_mod2.extend([c] * len(mod_2_class_test_data))

        class_test_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                         size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
        test_data_mod1.append(class_test_data)
        test_labels_mod1.extend([c] * len(class_test_data))

    train_data = np.concatenate(train_data)
    test_data_mod1 = np.concatenate(test_data_mod1)
    return train_data, train_labels, test_data_mod1, test_labels_mod1, test_data_mod2, test_labels_mod2


def train_and_eval(n_train_samples_per_class, second_modality=None):
    test_scores_mod_1 = []
    test_scores_mod_2 = []
    for seed in range(10):
        train_fmri_betas, train_labels, test_data_mod1, test_labels_mod1, test_data_mod2, test_labels_mod2 = generate_dummy_fmri_data(n_train_samples_per_class,
                                                                                                seed=seed,
                                                                                                second_modality=second_modality)
        clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=args.l2_regularization_alpha))
        clf.fit(train_fmri_betas, train_labels)

        test_acc_mod1 = np.mean(clf.predict(test_data_mod1) == test_labels_mod1)
        test_scores_mod_1.append(test_acc_mod1)
        test_acc_mod2 = np.mean(clf.predict(test_data_mod2) == test_labels_mod2)
        test_scores_mod_2.append(test_acc_mod2)


    print(f"MEAN modality 1: {np.mean(test_scores_mod_1):.2f}\n")
    print(f"MEAN modality 2: {np.mean(test_scores_mod_2):.2f}\n")
    return test_scores_mod_1, test_scores_mod_2


def run(args):
    results = []

    condition = "BASELINE\nM1=x+GAUSS(0, sigma_1)"
    scores_baseline_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS)
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_baseline_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    # condition = "BASELINE\n(2xtrain_samples)"
    # scores_mod_1, scores_mod_2 = train_and_eval(2*N_TRAIN_SAMPLES_PER_CLASS)
    # print(condition)
    # for score in scores:
    #     results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "SAME STDDEV\nM2=x+GAUSS(0, sigma_1)"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "gauss_same_stddev")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "SMALLER STDDEV\nM2=x+GAUSS(0, 0.5*sigma_1)"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "gauss_smaller_stddev")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "HIGHER STDDEV\nM2=x+GAUSS(0, 2*sigma_1)"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "gauss_higher_stddev")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "OFFSET\nM2=x+GAUSS(1, sigma_1)"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "offset")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "ORTHOGONAL\nM2=orth(x)+GAUSS(0, sigma_1)"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "orthogonal")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "INVERSE\nM2=-1*x+GAUSS(0, sigma_1)"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "inverse")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    # condition = "EXACT INVERSE"
    # scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "exact_inverse")
    # print(condition)
    # for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
    #     results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "JUST NOISE\nM2=GAUSS(0, sigma_1)"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "just_noise")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "INDEPENDENT\nM2=y+GAUSS(0, sigma_1)"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "independent")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "HALF INDEPENDENT,\nHALF SAME"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "half_independent_half_same")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "HALF NOISE,\nHALF SAME"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "half_noise_half_same")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    condition = "QUARTER NOISE,\nTHREE QUARTERS SAME"
    scores_mod_1, scores_mod_2 = train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "quarter_noise_three_quarters_same")
    print(condition)
    for score_mod_1, score_mod_2 in zip(scores_mod_1, scores_mod_2):
        results.append({"condition": condition, "acc_mod_1": score_mod_1, "acc_mod_2": score_mod_2})

    results = pd.DataFrame.from_records(results)
    sns.barplot(data=results, y="condition", x="acc_mod_1")
    plt.axvline(x=np.mean(scores_baseline_mod_1), color='black', linestyle="--", linewidth=0.7)
    # plt.xticks(rotation=85)
    plt.tight_layout()
    plt.ylabel("")
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--l2-regularization-alpha", type=float, default=1000)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
