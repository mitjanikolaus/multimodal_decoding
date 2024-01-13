############################################
# Training multimodal linear brain decoders
# inputs can be of any modality
# outputs are uni-modal
############################################
import math
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist, cosine
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import pickle
from torchvision.transforms import Compose
from decoding_utils import get_distance_matrix, to_tensor, HyperParameters
from tqdm import trange
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import IMAGERY_SCENES, MODEL_FEATURES_FILES, FMRI_DATA_DIR

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

NUM_CV_SPLITS = 5
PATIENCE = 5

TRAINING_MODES = ['train', 'train_captions', 'train_images']

MAX_EPOCHS = 400
BATCH_SIZE = 2000

MAX_SAMPLES_EVAL_METRICS = 1000

# SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-07']
# SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
SUBJECTS = ['sub-01', 'sub-02']
# SUBJECTS = ['sub-05', 'sub-07']
# SUBJECTS = ['sub-03']

MODEL_NAMES = ['CLIP_L']
# MODEL_NAMES = ['CLIP_L', 'CLIP_V', 'BERT_LARGE', 'GPT2XL_AVG', 'VITL16_ENCODER', 'RESNET152_AVGPOOL']
# MODEL_NAMES = ['CLIP_L_PCA768', 'CLIP_V_PCA768', 'RESNET152_AVGPOOL_PCA768']  # RESNET152_AVGPOOL_PCA768

TRAINING_MODE = TRAINING_MODES[0]
DECODER_TESTING_MODE = ['test', 'test_captions', 'test_images'][0]

TWO_STAGE_GLM_DATA_DIR = os.path.join(FMRI_DATA_DIR, "glm_manual/two-stage-mni/")

GLM_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/glm/")
DISTANCE_METRICS = ['cosine']

REGRESSION_MODEL_SKLEARN = "sklearn"
REGRESSION_MODEL_PYTORCH = "pytorch"
REGRESSION_MODEL = REGRESSION_MODEL_SKLEARN

HPs = [
    # HyperParameters(alpha=1),
    # HyperParameters(alpha=1e1),
    # HyperParameters(alpha=1e2),
    # HyperParameters(alpha=1e3),
    HyperParameters(alpha=1e4),
    HyperParameters(alpha=1e5),
    HyperParameters(alpha=1e6),
    # HyperParameters(alpha=1e7),

    # HyperParameters(optim_type='ADAMW', lr=0.001, wd=0, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAM', lr=0.0001, wd=0, dropout=False, loss='MSE'),

    # HyperParameters(optim_type='ADAM', lr=0.0001, wd=0.1, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAMW', lr=0.001, wd=0.1, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAM', lr=0.01, wd=0.1, dropout=False, loss='MSE'),

    # HyperParameters(optim_type='ADAM', lr=0.0001, wd=1, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAMW', lr=0.001, wd=1, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAM', lr=0.01, wd=1, dropout=False, loss='MSE'),

    # HyperParameters(optim_type='ADAM', lr=0.0001, wd=10, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAM', lr=0.001, wd=10, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAMW', lr=0.001, wd=10, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAMW', lr=0.001, wd=1000, dropout=False, loss='MSE'),
    #
    # HyperParameters(optim_type='ADAMW', lr=0.0001, wd=10, dropout=False, loss='MSE'),
    # HyperParameters(optim_type='ADAMW', lr=0.0001, wd=1000, dropout=False, loss='MSE'),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def fetch_coco_image(sid, coco_images_dir):
    r"""
    Helper function to load an image from the COCO dataset given its ID.

    Args:
        `sid` (int): Image ID
        `coco_images_dir` (str): Address to the COCO root directory 
    """
    file_name = f"{sid:012d}.jpg"
    image_add = list(glob(os.path.join(coco_images_dir, '*', file_name)))
    return plt.imread(image_add[0])


class LinearNet(nn.Module):
    r"""
    Simple linear network for Ridge regression.
    """

    def __init__(self, input_size, output_size, dropout=False):
        r"""
        Args:
            `input_size` (int): Size of the input vectors
            `output_size` (int): Size of the output vectors
            `dropout` (boolean): if `True`, a dropout module with probability 0.5 will be applied to the input
        """
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.drop = None
        if dropout:
            self.drop = nn.Dropout()

    def forward(self, x):
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)
        return x


class COCOBOLDDataset(Dataset):
    r"""
    Dataset for loading SEMREPS BOLD signals
    """

    def __init__(self, bold_root_dir, subject, model_name, mean_std_dir, mode=TRAINING_MODES[0],
                 blank_correction=True, subset=None, fold=None, preloaded_betas=None,
                 overwrite_transformations_mean_std=False,
                 fmri_betas_transform=None, nn_latent_transform=None, transform_to_tensor=False):
        """
        Args:
            bold_root_dir (str): BOLD root directory (parent of subjects' directories).
            subject (str): Subject ID.
            model_name (str): model name
            mode (str): 'train', 'test', or 'imagery'. You can append _images or _captions to make it unimodal
            blank_correction (boolean): If `True`, the blank image will be subtracted from the imagery patterns (if exists)
        """
        self.root_dir = os.path.join(bold_root_dir, subject)
        self.subject = subject
        self.imagery_scenes = IMAGERY_SCENES[subject]
        self.mode = mode
        self.blank = None
        self.blank_correction = blank_correction
        self.feature_key = ""
        self.mean_std_dir = mean_std_dir
        self.subset = subset
        self.overwrite_transformations_mean_std = overwrite_transformations_mean_std
        self.model_name = model_name
        self.fold = fold
        self.transform_to_tensor = transform_to_tensor

        latent_vectors_file = MODEL_FEATURES_FILES[model_name]

        latent_vectors = pickle.load(open(latent_vectors_file, 'rb'))
        for _, vec in latent_vectors.items():
            for key in vec:
                if 'feature' in key:
                    self.feature_key = key
                    break
            break
        if self.feature_key == "":
            raise Exception('no feature found!')

        self.fmri_betas_addresses = np.array(
            (sorted(glob(os.path.join(self.root_dir, f'betas_{self.mode}*', '*.nii')))))
        self.stim_ids = []
        self.stim_types = []
        self.nn_latent_vectors = []

        for addr in self.fmri_betas_addresses:
            file_name = os.path.basename(addr)
            if 'I' in file_name:  # Image
                stim_id = int(file_name[file_name.find('I') + 1:-4])
                self.stim_types.append('image')
            elif 'C' in file_name:  # Caption
                stim_id = int(file_name[file_name.find('C') + 1:-4])
                self.stim_types.append('caption')
            else:  # imagery
                stim_id = int(file_name[file_name.find('.nii') - 1:-4])
                stim_id = self.imagery_scenes[stim_id - 1][1]
                self.stim_types.append('imagery')
            self.stim_ids.append(stim_id)
            self.nn_latent_vectors.append(latent_vectors[stim_id][self.feature_key])

        self.nn_latent_vectors = np.array(self.nn_latent_vectors, dtype=np.float32)
        self.stim_ids = np.array(self.stim_ids)
        self.stim_types = np.array(self.stim_types)

        if preloaded_betas is not None:
            assert len(preloaded_betas) == len(self.fmri_betas_addresses), f"Preloaded betas shape does not match!"
            self.fmri_betas = preloaded_betas
        else:
            self.fmri_betas = np.array([None for _ in range(len(self.fmri_betas_addresses))])

        brain_mask_address = os.path.join(self.root_dir, f'unstructured', 'mask.nii')
        self.brain_mask = nib.load(brain_mask_address).get_fdata().reshape(-1)
        self.brain_mask = np.logical_and(np.logical_not(np.isnan(self.brain_mask)), self.brain_mask != 0)
        self.bold_dim_size = self.brain_mask.sum()
        self.latent_dim_size = self.nn_latent_vectors[0].shape[0]

        beta_blank_address = os.path.join(self.root_dir, f'betas_blank', 'beta_blank.nii')
        if os.path.exists(beta_blank_address):
            self.blank = nib.load(beta_blank_address).get_fdata().astype('float32').reshape(-1)
            self.blank = self.blank[self.brain_mask]

        if self.subset is not None:
            self.fmri_betas_addresses = self.fmri_betas_addresses[subset]
            self.fmri_betas = self.fmri_betas[subset]
            self.nn_latent_vectors = self.nn_latent_vectors[subset]
            self.stim_ids = self.stim_ids[subset]
            self.stim_types = self.stim_types[subset]

        self.fmri_betas_transform = fmri_betas_transform
        self.nn_latent_transform = nn_latent_transform

        if self.fmri_betas_transform is None:
            self.init_fmri_betas_transform()

        if self.nn_latent_transform is None:
            self.init_nn_latent_transform()

    def init_nn_latent_transform(self):
        model_std_mean_name = f'{self.model_name}_mean_std_{self.mode}_fold_{self.fold}.p'
        model_std_mean_path = os.path.join(self.mean_std_dir, model_std_mean_name)
        if self.overwrite_transformations_mean_std or (not os.path.exists(model_std_mean_path)):
            print(f"Calculating Mean and STD of Model Latent Variables for {self.mode} samples (fold: {self.fold})")
            os.makedirs(self.mean_std_dir, exist_ok=True)

            mean_std = {'mean': self.nn_latent_vectors.mean(axis=0),
                        'std': self.nn_latent_vectors.std(axis=0)}
            pickle.dump(mean_std, open(model_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        model_mean_std = pickle.load(open(model_std_mean_path, 'rb'))
        compose_ops = [Normalize(model_mean_std['mean'], model_mean_std['std'])]
        if self.transform_to_tensor:
            compose_ops.append(to_tensor)
        self.nn_latent_transform = Compose(compose_ops)

    def init_fmri_betas_transform(self):
        bold_std_mean_name = f'bold_multimodal_mean_std_{self.mode}_fold_{self.fold}.p'
        bold_std_mean_path = os.path.join(self.mean_std_dir, bold_std_mean_name)

        if self.overwrite_transformations_mean_std or (not os.path.exists(bold_std_mean_path)):
            print(f"Calculating Mean and STD of BOLD Signals for {self.mode} samples (fold: {self.fold})")
            os.makedirs(self.mean_std_dir, exist_ok=True)
            self.preload()

            mean_std = {'mean': self.fmri_betas.mean(axis=0),
                        'std': self.fmri_betas.std(axis=0)}
            pickle.dump(mean_std, open(bold_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        bold_mean_std = pickle.load(open(bold_std_mean_path, 'rb'))
        compose_ops = [Normalize(bold_mean_std['mean'], bold_mean_std['std'])]
        if self.transform_to_tensor:
            compose_ops.append(to_tensor)
        self.fmri_betas_transform = Compose(compose_ops)

    def preload(self):
        print("preloading")
        for idx in trange(len(self.fmri_betas_addresses)):
            if self.fmri_betas[idx] is None:
                sample = nib.load(self.fmri_betas_addresses[idx]).get_fdata().astype('float32').reshape(-1)
                sample = sample[self.brain_mask]
                self.fmri_betas[idx] = sample.copy()
        return self.fmri_betas

    def get_latent_vector(self, idx):
        latent = self.nn_latent_vectors[idx]
        if self.nn_latent_transform is not None:
            latent = self.nn_latent_transform(latent)
        return latent

    def get_brain_vector(self, idx):
        if self.fmri_betas[idx] is not None:
            sample = self.fmri_betas[idx]
        else:
            sample = nib.load(self.fmri_betas_addresses[idx]).get_fdata().astype('float32').reshape(-1)
            sample = sample[self.brain_mask]
            self.fmri_betas[idx] = sample.copy()

        if self.mode == 'imagery' and self.blank is not None and self.blank_correction:
            sample = sample - self.blank
        if self.fmri_betas_transform is not None:
            sample = self.fmri_betas_transform(sample)
        return sample

    def get_stim_id(self, idx):
        sid = self.stim_ids[idx]
        return sid

    def get_stim_type(self, idx):
        return self.stim_types[idx]

    def __len__(self):
        return len(self.fmri_betas_addresses)

    def __getitem__(self, idx):
        sample = self.get_brain_vector(idx)
        sid = self.get_stim_id(idx)
        latent_vector = self.get_latent_vector(idx)
        stype = self.get_stim_type(idx)

        return sample, latent_vector, sid, stype


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).astype(np.float32).squeeze()


class CosineDistance(nn.CosineSimilarity):
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__(dim, eps)

    def forward(self, x1, x2):
        return (1 - nn.functional.cosine_similarity(x1, x2, self.dim, self.eps)).mean()


def calc_cosine_distances(predictions, targets):
    return np.mean([cosine(pred, target) for pred, target in zip(predictions, targets)])


def train_decoder_epoch(model, train_loader, optimizer, loss_fn):
    model.train()
    cum_loss = []
    num_samples = 0
    for i, data in enumerate(train_loader, 0):
        inputs, latents, ids, types = data
        num_samples += inputs.shape[0]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, latents.to(device))
        cum_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    cum_loss = np.mean(cum_loss)
    return cum_loss, num_samples


def get_distance_matrix_csls(predictions, latents, knn=100, metric="cosine"):
    def get_nn_avg_dist(lat1, lat2, knn=10, metric="cosine"):
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


def pairwise_accuracy(predictions, latents, stimulus_ids, metric="cosine"):
    std = predictions.std(axis=0) + 1e-8  # For numerical stability
    predictions = (predictions - predictions.mean(axis=0)) / std

    if "csls_" in metric:
        metric = metric.replace("csls_", "")
        dist_mat = get_distance_matrix_csls(predictions, latents, metric=metric)
    else:
        dist_mat = get_distance_matrix(predictions, latents, metric)

    not_same_id = cdist(stimulus_ids.reshape(-1, 1), stimulus_ids.reshape(-1, 1)) != 0

    diag = dist_mat.diagonal().reshape(-1, 1)  # all congruent distances
    comp_mat = diag < dist_mat  # we are interested in i,j where d(i,i) < d(i,j)

    # Take only cases where the stimulus ids are not the same (do not compare cases where caption id == image id)
    comp_mat = comp_mat[not_same_id]
    score = comp_mat.mean()

    return score


def create_dissimilarity_matrix(sample_embeds):
    sim_mat = spearmanr(sample_embeds, axis=1)[0]
    dissim_mat = np.ones(sim_mat.shape) - sim_mat
    matrix = dissim_mat[np.triu_indices(sample_embeds.shape[0], 1)].reshape(-1)
    return matrix


def calc_rsa(latent_1, latent_2):
    matrix_1 = create_dissimilarity_matrix(latent_1)
    matrix_2 = create_dissimilarity_matrix(latent_2)
    corr = spearmanr([matrix_1, matrix_2], axis=1)[0]
    return corr


def calculate_eval_metrics(results):
    # take equally sized subsets of samples for captions and images
    stimulus_ids_caption = results["stimulus_ids"][results["stimulus_types"] == 'caption'][:MAX_SAMPLES_EVAL_METRICS]
    stimulus_ids_image = results["stimulus_ids"][results["stimulus_types"] != 'caption'][:MAX_SAMPLES_EVAL_METRICS]
    val_ids = np.concatenate((stimulus_ids_caption, stimulus_ids_image))

    predictions_caption = results["predictions"][results["stimulus_types"] == 'caption'][:MAX_SAMPLES_EVAL_METRICS]
    predictions_image = results["predictions"][results["stimulus_types"] != 'caption'][:MAX_SAMPLES_EVAL_METRICS]
    val_predictions = np.concatenate((predictions_caption, predictions_image))

    latents_caption = results["latents"][results["stimulus_types"] == 'caption'][:MAX_SAMPLES_EVAL_METRICS]
    latents_image = results["latents"][results["stimulus_types"] != 'caption'][:MAX_SAMPLES_EVAL_METRICS]
    val_latents = np.concatenate((latents_caption, latents_image))

    for metric in DISTANCE_METRICS:
        acc = pairwise_accuracy(val_predictions, val_latents, val_ids, metric)
        results[f"acc_{metric}"] = acc

        acc_captions = pairwise_accuracy(predictions_caption, latents_caption, stimulus_ids_caption, metric)
        acc_images = pairwise_accuracy(predictions_image, latents_image, stimulus_ids_image, metric)
        results[f"acc_{metric}_captions"] = acc_captions
        results[f"acc_{metric}_images"] = acc_images

    rsa = calc_rsa(val_predictions, val_latents)
    results['rsa'] = rsa

    return results


def evaluate_decoder(model, test_loader, loss_fn, return_preds=False):
    r"""
    evaluates decoder on test bold signals
    returns the predicted vectors and loss values
    `distance_metrics` is a list of string containing distance metric names
    """
    model.eval()
    loss = []
    predictions = []
    latents = []

    stimulus_ids = []
    stimulus_types = []
    with torch.no_grad():
        for data in test_loader:
            inputs, targets, ids, types = data
            preds = model(inputs.to(device))
            test_loss = loss_fn(preds, targets.to(device))
            loss.append(test_loss.item())
            if return_preds:
                predictions.append(preds.cpu().numpy())
                stimulus_ids.append(ids.cpu().numpy())
                stimulus_types.append(types)
                latents.append(targets.cpu().numpy())
    loss = np.mean(loss)
    results = {"loss": loss}

    if return_preds:
        predictions = np.concatenate(predictions, axis=0)
        stimulus_ids = np.concatenate(stimulus_ids, axis=0)
        stimulus_types = np.concatenate(stimulus_types, axis=0)
        latents = np.concatenate(latents, axis=0)

        results.update({'stimulus_ids': stimulus_ids,
                        'stimulus_types': stimulus_types,
                        'predictions': predictions,
                        'latents': latents,
                        'loss': loss})

    return results


def create_optimizer(hp, model):
    if hp.optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=hp.lr, weight_decay=hp.wd)
    elif hp.optim_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)
    elif hp.optim_type == 'ADAMW':
        optimizer = optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.wd)
    else:
        raise RuntimeError("Unknown optimizer: ", hp.optim_type)
    return optimizer


def train_and_test(hp, run_str, results_dir, train_ds, val_ds=None, test_ds=None, max_samples=None):
    results_file_dir = f'{results_dir}/{run_str}'
    checkpoint_dir = f'{results_dir}/networks/{run_str}'
    os.makedirs(results_file_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if REGRESSION_MODEL == REGRESSION_MODEL_PYTORCH:
        loss_fn = nn.MSELoss() if hp.loss_type == 'MSE' else CosineDistance()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=True,
                                       drop_last=True)
        if val_ds is not None:
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
        if test_ds is not None:
            test_loader = DataLoader(test_ds, batch_size=len(test_dataset), num_workers=0, shuffle=False)

        model = LinearNet(train_ds.bold_dim_size, train_ds.latent_dim_size, dropout=hp.dropout)
        model = model.to(device)
        optimizer = create_optimizer(hp, model)

        sumwriter = SummaryWriter(f'{results_dir}/tensorboard/{run_str}', filename_suffix=f'')
        epochs_no_improved_loss = 0
        best_val_loss = math.inf
        best_val_loss_num_samples = 0
        num_samples_train_run = 0
        for _ in trange(MAX_EPOCHS, desc=f'training decoder'):
            train_loss, num_epoch_samples = train_decoder_epoch(model, train_loader, optimizer, loss_fn)
            num_samples_train_run += num_epoch_samples
            sumwriter.add_scalar(f"Training/{hp.loss_type} loss", train_loss, num_samples_train_run)

            if val_ds is not None:
                val_results = evaluate_decoder(model, val_loader, loss_fn)
                sumwriter.add_scalar(f"Val/{hp.loss_type} loss", val_results['loss'], num_samples_train_run)
                if val_results['loss'] < best_val_loss:
                    best_val_loss = val_results['loss']
                    best_val_loss_num_samples = num_samples_train_run
                    epochs_no_improved_loss = 0

                    torch.save(model.state_dict(), f"{checkpoint_dir}/model_best_val.pt")
                else:
                    epochs_no_improved_loss += 1

                if epochs_no_improved_loss >= PATIENCE:
                    print(f"Loss did not improve for {PATIENCE} epochs. Terminating training.")
                    break

            elif max_samples is not None:
                if num_samples_train_run >= max_samples:
                    print(f"reached {max_samples} samples. Terminating full train.")

                    torch.save(model.state_dict(), f"{checkpoint_dir}/model_best_val.pt")
                    break
            else:
                raise RuntimeError("Need to specify either a val loader or max_samples for train loop!")

        sumwriter.close()

        # Final eval
        model = LinearNet(train_ds.bold_dim_size, train_ds.latent_dim_size, dropout=hp.dropout)
        model = model.to(device)
        model.load_state_dict(torch.load(f"{checkpoint_dir}/model_best_val.pt", map_location=device))

        results = {"best_val_loss_num_samples": best_val_loss_num_samples}

        if val_ds is not None:
            val_results = evaluate_decoder(model, val_loader, loss_fn, return_preds=True)
            val_results = calculate_eval_metrics(val_results)
            results = {**results, **{'val_' + key: val for key, val in val_results.items()}}
        if test_ds is not None:
            test_results = evaluate_decoder(model, test_loader, loss_fn, return_preds=True)
            test_results = calculate_eval_metrics(test_results)
            results = {**results, **test_results}

    elif REGRESSION_MODEL == REGRESSION_MODEL_SKLEARN:
        loss_fn = nn.MSELoss() if hp.loss_type == 'MSE' else calc_cosine_distances

        train_data = [d for d in train_ds]
        train_data_inputs = np.array([i for i, _, _, _ in train_data])
        train_data_latents = np.array([l for _, l, _, _ in train_data])

        model = Ridge(alpha=hp.alpha)
        model.fit(train_data_inputs, train_data_latents)

        results = {"best_val_loss_num_samples": 0}
        if val_ds is not None:
            val_data = [d for d in val_ds]
            val_data_inputs = np.array([i for i, _, _, _ in val_data])
            val_data_latents = np.array([l for _, l, _, _ in val_data])

            val_predicted_latents = model.predict(val_data_inputs)
            val_loss = loss_fn(val_predicted_latents, val_data_latents)
            val_results = {"loss": val_loss,
                           "stimulus_ids": val_ds.stim_ids,
                           "stimulus_types": val_ds.stim_types,
                           "predictions": val_predicted_latents,
                           "latents": val_data_latents}
            val_results = calculate_eval_metrics(val_results)
            results = {**results, **{'val_' + key: val for key, val in val_results.items()}}

        if test_ds is not None:
            test_data = [d for d in test_ds]
            test_data_inputs = np.array([i for i, _, _, _ in test_data])
            test_data_latents = np.array([l for _, l, _, _ in test_data])
            test_predicted_latents = model.predict(test_data_inputs)

            test_results = {"stimulus_ids": test_ds.stim_ids,
                            "stimulus_types": test_ds.stim_types,
                            "predictions": test_predicted_latents,
                            "latents": test_data_latents}
            test_results = calculate_eval_metrics(test_results)
            results = {**results, **test_results}
    else:
        raise RuntimeError("Unknown regression model: ", REGRESSION_MODEL)

    pickle.dump(results, open(os.path.join(results_file_dir, "results.p"), 'wb'))

    return results


def retrain_full_train(train_dataset, test_dataset, hp_setting, num_samples, results_dir, suffix="_full_train"):
    num_samples_per_epoch = (len(train_dataset) // BATCH_SIZE) * BATCH_SIZE
    best_hp_setting_num_epochs = (num_samples // num_samples_per_epoch) + 1
    print(
        f"Retraining for {best_hp_setting_num_epochs} epochs on full train set with hp setting: ",
        hp_setting.to_string())


    run_str = hp_setting.to_string() + suffix

    results = train_and_test(hp_setting, run_str, results_dir, train_dataset, val_ds=None,
                             test_ds=test_dataset,
                             max_samples=num_samples)

    best_dir = f'{results_dir}/best_hp/'
    os.makedirs(best_dir, exist_ok=True)
    pickle.dump(results, open(os.path.join(best_dir, "results.p"), 'wb'))


if __name__ == "__main__":
    print("device: ", device)
    os.makedirs(GLM_OUT_DIR, exist_ok=True)

    for subject in SUBJECTS:
        print(subject)
        for model_name in MODEL_NAMES:
            print(model_name)
            results_dir = os.path.join(GLM_OUT_DIR, f'{TRAINING_MODE}/{REGRESSION_MODEL}/{subject}/{model_name}')

            std_mean_dir = os.path.join(GLM_OUT_DIR, subject)

            transform_to_tensor = True if REGRESSION_MODEL == REGRESSION_MODEL_PYTORCH else False
            train_val_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                TRAINING_MODE, transform_to_tensor=transform_to_tensor)
            preloaded_betas = train_val_dataset.preload()

            kf = KFold(n_splits=NUM_CV_SPLITS, shuffle=True, random_state=1)

            test_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                           DECODER_TESTING_MODE,
                                           fmri_betas_transform=train_val_dataset.fmri_betas_transform,
                                           nn_latent_transform=train_val_dataset.nn_latent_transform,
                                           transform_to_tensor=transform_to_tensor)
            test_dataset.preload()

            best_hp_setting = None
            best_hp_setting_val_loss = math.inf
            best_hp_setting_num_samples = None

            best_hp_setting_acc_cosine = None
            best_hp_setting_acc_cosine_value = 0
            best_hp_setting_acc_cosine_num_samples = None

            for hp in HPs:
                print(hp.to_string())

                val_losses_for_folds = []
                num_samples_for_folds = []

                accs_cosine_for_folds = []

                start = time.time()

                for fold, (train_idx, val_idx) in enumerate(kf.split(list(range(len(train_val_dataset))))):
                    train_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                    TRAINING_MODE, subset=train_idx, fold=fold,
                                                    preloaded_betas=preloaded_betas,
                                                    transform_to_tensor=transform_to_tensor)
                    val_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                  TRAINING_MODE, subset=val_idx, fold=fold,
                                                  preloaded_betas=preloaded_betas,
                                                  fmri_betas_transform=train_dataset.fmri_betas_transform,
                                                  nn_latent_transform=train_dataset.nn_latent_transform,
                                                  transform_to_tensor=transform_to_tensor
                                                  )
                    print(f"Fold {fold} | train set size: {len(train_dataset)} | val set size: {len(val_dataset)}")

                    run_str = hp.to_string() + f"fold_{fold}"

                    results = train_and_test(hp, run_str, results_dir, train_dataset, val_dataset, test_dataset)

                    val_losses_for_folds.append(results['val_loss'])
                    num_samples_for_folds.append(results["best_val_loss_num_samples"])
                    print(f"best val loss: {results['val_loss']:.4f}")

                    accs_cosine_for_folds.append(results['val_acc_cosine'])
                    print(f"best val acc cosine: {results['val_acc_cosine']:.4f}")

                if np.mean(val_losses_for_folds) < best_hp_setting_val_loss:
                    best_hp_setting_val_loss = np.mean(val_losses_for_folds)
                    best_hp_setting = hp
                    best_hp_setting_num_samples = int(np.mean(num_samples_for_folds))
                    print(
                        f"new best hp setting val loss: {np.mean(val_losses_for_folds):.4f} | "
                        f"num samples: {best_hp_setting_num_samples}")

                if np.mean(accs_cosine_for_folds) > best_hp_setting_acc_cosine_value:
                    best_hp_setting_acc_cosine_value = np.mean(accs_cosine_for_folds)
                    best_hp_setting_acc_cosine = hp
                    best_hp_setting_acc_cosine_num_samples = int(np.mean(num_samples_for_folds))
                    print(
                        f"new best hp setting acc cosine: {np.mean(accs_cosine_for_folds):.4f} | "
                        f"num samples: {best_hp_setting_acc_cosine_num_samples}")

                end = time.time()
                print(f"Elapsed time: {int(end - start)}s")

            # Re-train on full train set with best HP settings:
            retrain_full_train(train_val_dataset, test_dataset, best_hp_setting, best_hp_setting_num_samples,
                               results_dir)

            retrain_full_train(train_val_dataset, test_dataset, best_hp_setting_acc_cosine,
                               best_hp_setting_acc_cosine_num_samples, results_dir, suffix="_full_train_best_val_acc")
