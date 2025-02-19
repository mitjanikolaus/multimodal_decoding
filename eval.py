import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler

from data import CAPTION, IMAGE

ACC_MODALITY_AGNOSTIC = "pairwise_acc_modality_agnostic"
ACC_CAPTIONS = "pairwise_acc_captions"
ACC_IMAGES = "pairwise_acc_images"

ACC_CROSS_IMAGES_TO_CAPTIONS = "pairwise_acc_cross_images_to_captions"
ACC_CROSS_CAPTIONS_TO_IMAGES = "pairwise_acc_cross_captions_to_images"

ACC_IMAGES_MOD_AGNOSTIC = "pairwise_acc_images_mod_agnostic"
ACC_CAPTIONS_MOD_AGNOSTIC = "pairwise_acc_captions_mod_agnostic"
ACC_IMAGERY_MOD_AGNOSTIC = "pairwise_acc_imagery_mod_agnostic"
ACC_IMAGERY_WHOLE_TEST_SET_MOD_AGNOSTIC = "pairwise_acc_imagery_whole_test_set_mod_agnostic"

ACC_IMAGERY_NO_STD_MOD_AGNOSTIC = "pairwise_acc_imagery_no_std_mod_agnostic"
ACC_IMAGERY_WHOLE_TEST_SET_NO_STD_MOD_AGNOSTIC = "pairwise_acc_imagery_whole_test_set_no_std_mod_agnostic"

ACC_IMAGES_MOD_SPECIFIC_IMAGES = "pairwise_acc_images_mod_specific_images"
ACC_CAPTIONS_MOD_SPECIFIC_IMAGES = "pairwise_acc_captions_mod_specific_images"

ACC_CAPTIONS_MOD_SPECIFIC_CAPTIONS = "pairwise_acc_captions_mod_specific_captions"
ACC_IMAGES_MOD_SPECIFIC_CAPTIONS = "pairwise_acc_images_mod_specific_captions"

ACC_IMAGERY = "pairwise_acc_imagery"
ACC_IMAGERY_WHOLE_TEST = "pairwise_acc_imagery_whole_test_set"

CORR_ALL = "corr"
CORR_CAPTIONS = "corr_captions"
CORR_IMAGES = "corr_images"
CORR_CROSS_IMAGES_TO_CAPTIONS = "corr_cross_images_to_captions"
CORR_CROSS_CAPTIONS_TO_IMAGES = "corr_cross_captions_to_images"

CORR_CAPTIONS_MOD_AGNOSTIC = "corr_captions_mod_agnostic"
CORR_IMAGES_MOD_AGNOSTIC = "corr_images_mod_agnostic"
CORR_CAPTIONS_MOD_SPECIFIC_CAPTIONS = "corr_captions_mod_specific_captions"
CORR_IMAGES_MOD_SPECIFIC_CAPTIONS = "corr_images_mod_specific_captions"
CORR_IMAGES_MOD_SPECIFIC_IMAGES = "corr_images_mod_specific_images"
CORR_CAPTIONS_MOD_SPECIFIC_IMAGES = "corr_captions_mod_specific_images"

METRIC_CROSS_ENCODING = "corr_cross_encoding"

DISTANCE_METRIC_COSINE = "cosine"


def get_distance_matrix_csls(predictions, latents, knn=100, metric=DISTANCE_METRIC_COSINE):
    def get_nn_avg_dist(lat1, lat2, knn, metric):
        distances = cdist(lat2, lat1, metric=metric)

        best_distances_idx = np.argsort(distances, axis=1)[:, -knn:]
        best_distances = distances[best_distances_idx]

        all_distances = best_distances.mean(axis=1)

        return all_distances

    average_dist_preds = get_nn_avg_dist(predictions, latents, knn, metric)
    average_dist_lats = get_nn_avg_dist(latents, predictions, knn, metric)

    scores = cdist(predictions, latents, metric=metric)

    dist_mat = 2 * scores - average_dist_preds - average_dist_lats

    return dist_mat


def get_distance_matrix(predictions, originals, metric='cosine'):
    return cdist(predictions, originals, metric=metric)


def dist_mat_to_pairwise_acc(dist_mat):
    diag = dist_mat.diagonal().reshape(-1, 1)
    comp_mat = diag < dist_mat
    corrects = comp_mat.sum()
    # subtract the number of elements of the diagonal as these values are always "False" (not smaller than themselves)
    score = corrects / (dist_mat.size - diag.size)
    return score


def pairwise_accuracy(latents, predictions, metric="cosine", standardize_predictions=False, standardize_latents=False):
    if standardize_predictions:
        predictions = StandardScaler().fit_transform(predictions)
    if standardize_latents:
        latents = StandardScaler().fit_transform(latents)

    dist_mat = get_distance_matrix(predictions, latents, metric)
    return dist_mat_to_pairwise_acc(dist_mat)


def calc_all_pairwise_accuracy_scores(latents, predictions, stim_types=None, imagery_latents=None,
                                      imagery_predictions=None, metric="cosine", standardize_predictions=True,
                                      standardize_latents=False, norm_imagery_preds_with_test_preds=False,
                                      comp_cross_decoding_scores=True):
    results = dict()
    for modality, acc_metric_name in zip([CAPTION, IMAGE], [ACC_CAPTIONS, ACC_IMAGES]):
        preds_mod = predictions[stim_types == modality]
        latents_mod = latents[stim_types == modality]

        results[acc_metric_name] = pairwise_accuracy(
            latents_mod, preds_mod, metric, standardize_predictions, standardize_latents
        )

    if comp_cross_decoding_scores:
        for mod_preds, mod_latents, acc_metric_name in zip([CAPTION, IMAGE], [IMAGE, CAPTION],
                                                           [ACC_CROSS_CAPTIONS_TO_IMAGES,
                                                            ACC_CROSS_IMAGES_TO_CAPTIONS]):
            preds_mod = predictions[stim_types == mod_preds]
            latents_mod = latents[stim_types == mod_latents]

            results[acc_metric_name] = pairwise_accuracy(
                latents_mod, preds_mod, metric, standardize_predictions, standardize_latents
            )

    if imagery_latents is not None:
        imagery_scores = calc_imagery_pairwise_accuracy_scores(
            imagery_latents, imagery_predictions, latents, metric,
            standardize_predictions, standardize_latents,
            test_set_preds=predictions if norm_imagery_preds_with_test_preds else None
        )
        results.update(imagery_scores)

    return results


def calc_imagery_pairwise_accuracy_scores(imagery_latents, imagery_predictions, additional_latents, metric="cosine",
                                          standardize_predictions=False, standardize_latents=False, test_set_preds=None):
    results = dict()

    if test_set_preds is not None:
        all_preds = np.concatenate((imagery_predictions, test_set_preds))
        scaler = StandardScaler().fit(all_preds)
        imagery_predictions = scaler.transform(imagery_predictions)

        standardize_predictions = False  # Do not standardize again

    results[ACC_IMAGERY] = pairwise_accuracy(
        imagery_latents, imagery_predictions, metric, standardize_predictions, standardize_latents
    )

    target_latents = np.concatenate((imagery_latents, additional_latents))
    results[ACC_IMAGERY_WHOLE_TEST] = pairwise_accuracy(
        target_latents, imagery_predictions, metric, standardize_predictions, standardize_latents
    )

    return results


def create_dissimilarity_matrix(sample_embeds, matrix_metric="spearmanr"):
    if matrix_metric == "spearmanr":
        sim_mat = spearmanr(sample_embeds, axis=1)[0]
    elif matrix_metric == "pearsonr":
        sim_mat = np.corrcoef(sample_embeds, rowvar=1)
    else:
        raise RuntimeError("Unknown metric: ", matrix_metric)
    dissim_mat = np.ones(sim_mat.shape) - sim_mat
    matrix = dissim_mat[np.triu_indices(sample_embeds.shape[0], 1)].reshape(-1)
    return matrix


def rsa_from_matrices(matrix_1, matrix_2, metric="spearmanr"):
    if metric == "spearmanr":
        corr = spearmanr([matrix_1, matrix_2], axis=1)[0]
    elif metric == "pearsonr":
        corr = pearsonr(matrix_1, matrix_2)[0]
    else:
        raise RuntimeError("Unknown metric: ", metric)
    return corr


def calc_rsa(latent_1, latent_2, metric="spearmanr", matrix_metric="spearmanr"):
    matrix_1 = create_dissimilarity_matrix(latent_1, matrix_metric)
    matrix_2 = create_dissimilarity_matrix(latent_2, matrix_metric)
    return rsa_from_matrices(matrix_1, matrix_2, metric=metric)


def calc_rsa_images(latent_1, latent_2, stimulus_types, metric="spearmanr", matrix_metric="spearmanr"):
    assert len(latent_1) == len(latent_2) == len(stimulus_types)
    latent_1_images = latent_1[stimulus_types == IMAGE]
    latent_2_images = latent_2[stimulus_types == IMAGE]
    return calc_rsa(latent_1_images, latent_2_images, metric, matrix_metric)


def calc_rsa_captions(latent_1, latent_2, stimulus_types, metric="spearmanr", matrix_metric="spearmanr"):
    assert len(latent_1) == len(latent_2) == len(stimulus_types)
    latent_1_captions = latent_1[stimulus_types == CAPTION]
    latent_2_captions = latent_2[stimulus_types == CAPTION]
    return calc_rsa(latent_1_captions, latent_2_captions, metric, matrix_metric)
