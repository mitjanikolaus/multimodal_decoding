import argparse

import numpy as np
from sklearn.linear_model import RidgeClassifier

N_CLASSES = 10
N_TRAIN_SAMPLES_PER_CLASS = 100
N_TEST_SAMPLES_PER_CLASS = 1000

N_VOXELS_FMRI = 100

STDDEV_WITHIN_CLASS = 1.5


def get_dummy_fmri_data(n_train_samples_per_class, seed, second_modality=None):
    np.random.seed(seed)
    data_classes = np.random.uniform(size=(N_CLASSES, N_VOXELS_FMRI))
    if second_modality is not None and second_modality == "independent":
        data_classes_mod_2 = np.random.uniform(size=(N_CLASSES, N_VOXELS_FMRI))

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for c, class_proto in enumerate(data_classes):
        class_train_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                          size=(n_train_samples_per_class, N_VOXELS_FMRI))
        train_data.append(class_train_data)
        train_labels.extend([c] * len(class_train_data))
        if second_modality is not None:
            if second_modality == "gauss_same_stddev":
                mod_2_class_train_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                mod_2_class_train_data += np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                           size=(n_train_samples_per_class, N_VOXELS_FMRI))
                train_data.append(mod_2_class_train_data)
                train_labels.extend([c] * len(mod_2_class_train_data))

            elif second_modality == "gauss_smaller_stddev":
                mod_2_class_train_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                mod_2_class_train_data += np.random.normal(scale=0.5*STDDEV_WITHIN_CLASS,
                                                           size=(n_train_samples_per_class, N_VOXELS_FMRI))
                train_data.append(mod_2_class_train_data)
                train_labels.extend([c] * len(mod_2_class_train_data))

            elif second_modality == "gauss_higher_stddev":
                mod_2_class_train_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                mod_2_class_train_data += np.random.normal(scale=2*STDDEV_WITHIN_CLASS,
                                                           size=(n_train_samples_per_class, N_VOXELS_FMRI))
                train_data.append(mod_2_class_train_data)
                train_labels.extend([c] * len(mod_2_class_train_data))

            elif second_modality == "inverse":
                mod_2_class_train_data = -1 * class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                train_data.append(mod_2_class_train_data)
                train_labels.extend([c] * len(mod_2_class_train_data))

            elif second_modality == "exact_inverse":
                mod_2_class_train_data = -1 * class_train_data
                train_data.append(mod_2_class_train_data)
                train_labels.extend([c] * len(mod_2_class_train_data))

            elif second_modality == "orthogonal":
                def get_orthogonal(k):
                    x = np.random.randn(k.shape[0])
                    x -= x.dot(k) * k
                    x /= np.linalg.norm(x)
                    return x
                mod_2_class_train_data = np.array([get_orthogonal(x) for x in class_train_data])
                train_data.append(mod_2_class_train_data)
                train_labels.extend([c] * len(mod_2_class_train_data))

            elif second_modality == "independent":
                mod_2_class_train_data = data_classes_mod_2[c] + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                                        size=(n_train_samples_per_class,
                                                                              N_VOXELS_FMRI))
                train_data.append(mod_2_class_train_data)
                train_labels.extend([c] * len(mod_2_class_train_data))

            elif second_modality == "just_noise":
                mod_2_class_train_data = np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                            size=(n_train_samples_per_class,
                                                                  N_VOXELS_FMRI))
                train_data.append(mod_2_class_train_data)
                train_labels.extend([c] * len(mod_2_class_train_data))

            else:
                raise RuntimeError("Unknown second modality option: ", second_modality)

        class_test_data = class_proto + np.random.normal(scale=STDDEV_WITHIN_CLASS,
                                                         size=(N_TEST_SAMPLES_PER_CLASS, N_VOXELS_FMRI))
        test_data.append(class_test_data)
        test_labels.extend([c] * len(class_test_data))

    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    return train_data, test_data, train_labels, test_labels


def train_and_eval(n_train_samples_per_class, second_modality=None):
    test_scores = []
    for seed in range(10):
        train_fmri_betas, test_fmri_betas, train_labels, test_labels = get_dummy_fmri_data(n_train_samples_per_class,
                                                                                           seed=seed,
                                                                                           second_modality=second_modality)
        # clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=args.l2_regularization_alpha))
        clf = RidgeClassifier(alpha=args.l2_regularization_alpha)

        clf.fit(train_fmri_betas, train_labels)

        test_predicted_labels = clf.predict(test_fmri_betas)
        test_acc = np.mean(test_predicted_labels == test_labels)
        # print(f"Test acc: {test_acc:.4f}")

        test_scores.append(test_acc)

    print(f"MEAN: {np.mean(test_scores):.2f}\n")


def run(args):
    print("BASELINE")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS)

    print("BASELINE (double training data)")
    train_and_eval(2*N_TRAIN_SAMPLES_PER_CLASS)

    print("SECOND MODALITY: INDEPENDENT")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "independent")

    print("SECOND MODALITY: GAUSSIAN NOISE (same stddev)")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "gauss_same_stddev")

    print("SECOND MODALITY: GAUSSIAN NOISE (smaller stddev)")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "gauss_smaller_stddev")

    print("SECOND MODALITY: GAUSSIAN NOISE (higher stddev)")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "gauss_higher_stddev")

    print("SECOND MODALITY: INVERSE")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "inverse")

    print("SECOND MODALITY: ORTHOGONAL")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "orthogonal")

    print("SECOND MODALITY: EXACT INVERSE")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "exact_inverse")

    print("SECOND MODALITY: JUST NOISE")
    train_and_eval(N_TRAIN_SAMPLES_PER_CLASS, "just_noise")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--l2-regularization-alpha", type=float, default=1000)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
