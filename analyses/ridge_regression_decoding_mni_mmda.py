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
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import pickle
from torchvision.transforms import Compose
from decoding_utils import get_distance_matrix, to_tensor
from tqdm import trange
import gc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import IMAGERY_SCENES, MODEL_FEATURES_FILES, FMRI_DATA_DIR

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

NUM_CV_SPLITS = 5
PATIENCE = 5

TRAINING_MODES = ['train', 'train_captions', 'train_images']


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
                 fmri_betas_transform=None, nn_latent_transform=None):
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

        self.nn_latent_vectors = np.array(self.nn_latent_vectors)
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

            mean_std = {'mean': self.nn_latent_vectors.mean(axis=0).astype('float32'),
                        'std': self.nn_latent_vectors.std(axis=0).astype('float32')}
            pickle.dump(mean_std, open(model_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        model_mean_std = pickle.load(open(model_std_mean_path, 'rb'))
        self.nn_latent_transform = Compose([
            Normalize(model_mean_std['mean'], model_mean_std['std']),
            to_tensor
        ])

    def init_fmri_betas_transform(self):
        bold_std_mean_name = f'bold_multimodal_mean_std_{self.mode}_fold_{self.fold}.p'
        bold_std_mean_path = os.path.join(self.mean_std_dir, bold_std_mean_name)

        if self.overwrite_transformations_mean_std or (not os.path.exists(bold_std_mean_path)):
            print(f"Calculating Mean and STD of BOLD Signals for {self.mode} samples (fold: {self.fold})")
            os.makedirs(self.mean_std_dir, exist_ok=True)
            self.preload()

            mean_std = {'mean': self.fmri_betas.mean(axis=0).astype('float32'),
                        'std': self.fmri_betas.std(axis=0).astype('float32')}
            pickle.dump(mean_std, open(bold_std_mean_path, 'wb'), pickle.HIGHEST_PROTOCOL)

        bold_mean_std = pickle.load(open(bold_std_mean_path, 'rb'))
        self.fmri_betas_transform = Compose([
            Normalize(bold_mean_std['mean'], bold_mean_std['std']),
            to_tensor
        ])

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


class HyperParameters():
    r"""
    A helper class to pack and represent training hyperparameters.
    """

    def __init__(self, optimizer='SGD', lr=0.01, wd=0.01, dropout=False, loss='MSE') -> None:
        self.optimizer = optimizer
        self.lr = lr
        self.wd = wd
        self.dropout = dropout
        self.loss = loss

    def get_hp_string(self):
        return f"[optim:{self.optimizer}][lr:{str(self.lr).replace('.', '-')}][wd:{str(self.wd).replace('.', '-')}][drop:{self.dropout}][loss:{self.loss}]"

    def __iter__(self):
        yield self.optimizer
        yield self.lr
        yield self.wd
        yield self.dropout
        yield self.loss


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


def pairwise_accuracy(predictions, latents, metric, stimulus_ids):
    predictions = (predictions - predictions.mean(axis=0)) / predictions.std(axis=0)

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


def evaluate_decoder(model, test_loader, loss_fn, calc_eval_metrics=False):
    r"""
    evaluates decoder on test bold signals
    returns the predicted vectors and loss values
    `distance_metrics` is a list of string containing distance metric names
    """
    model.eval()
    cum_loss = []
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs, latents, stimulus_ids, stimulus_types = data
            outputs = model(inputs.to(device))
            test_loss = loss_fn(outputs, latents.to(device))
            cum_loss.append(test_loss.item())
            predictions.append(outputs.cpu().numpy())
    cum_loss = np.mean(cum_loss)
    predictions = np.concatenate(predictions, axis=0)

    results = {'classes': stimulus_ids,
               'types': stimulus_types,
               'predictions': predictions,
               'latents': latents,
               'loss': cum_loss}

    if calc_eval_metrics:
        # take equally sized subsets of samples for captions and images
        stimulus_types = np.array(stimulus_types)

        stimulus_ids_caption = stimulus_ids[stimulus_types == 'caption'][:MAX_SAMPLES_EVAL_METRICS]
        stimulus_ids_image = stimulus_ids[stimulus_types != 'caption'][:MAX_SAMPLES_EVAL_METRICS]
        val_ids = np.concatenate((stimulus_ids_caption, stimulus_ids_image))

        predictions_caption = predictions[stimulus_types == 'caption'][:MAX_SAMPLES_EVAL_METRICS]
        predictions_image = predictions[stimulus_types != 'caption'][:MAX_SAMPLES_EVAL_METRICS]
        val_predictions = np.concatenate((predictions_caption, predictions_image))

        latents_caption = latents[stimulus_types == 'caption'][:MAX_SAMPLES_EVAL_METRICS]
        latents_image = latents[stimulus_types != 'caption'][:MAX_SAMPLES_EVAL_METRICS]
        val_latents = np.concatenate((latents_caption, latents_image))

        for metric in DISTANCE_METRICS:
            acc = pairwise_accuracy(val_predictions, val_latents, metric, val_ids)
            results[f"acc_{metric}"] = acc

            acc_captions = pairwise_accuracy(predictions_caption, latents_caption, metric, stimulus_ids_caption)
            acc_images = pairwise_accuracy(predictions_image, latents_image, metric, stimulus_ids_image)
            results[f"acc_{metric}_captions"] = acc_captions
            results[f"acc_{metric}_images"] = acc_images

        rsa = calc_rsa(val_predictions, val_latents)
        results['rsa'] = rsa

    return results


MAX_EPOCHS = 400
BATCH_SIZE = 2000

MAX_SAMPLES_EVAL_METRICS = 1000

# SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-05', 'sub-07']  # TODO 'sub-03'
SUBJECTS = ['sub-01', 'sub-02']

# model_names = ['GPT2XL_AVG', 'VITL16_ENCODER','RESNET152_AVGPOOL', 'GPT2XL_AVG_PCA768', 'VITL16_ENCODER_PCA768']
MODEL_NAMES = ['RESNET152_AVGPOOL']
# MODEL_NAMES = ['CLIP_L', 'CLIP_V', 'CLIP_L_PCA768', 'CLIP_V_PCA768', 'RESNET152_AVGPOOL']  # RESNET152_AVGPOOL_PCA768
# MODEL_NAMES = ['BERT_LARGE', 'CLIP_L', 'CLIP_V', 'VITL16_ENCODER', 'RESNET152_AVGPOOL', 'GPT2XL_AVG']
TRAINING_MODE = TRAINING_MODES[0]
DECODER_TESTING_MODE = ['test', 'test_captions', 'test_images'][0]

TWO_STAGE_GLM_DATA_DIR = os.path.join(FMRI_DATA_DIR, "glm_manual/two-stage-mni/")

GLM_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/glm/")
DISTANCE_METRICS = ['cosine', 'euclidean']

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_optimizer(optim_type):
    if optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, weight_decay=wd)
    elif optim_type == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise RuntimeError("Unknown optimizer: ", optim_type)
    return optimizer


if __name__ == "__main__":
    print("device: ", device)
    os.makedirs(GLM_OUT_DIR, exist_ok=True)

    for subject in SUBJECTS:
        print(subject)
        for model_name in MODEL_NAMES:
            print(model_name)
            std_mean_dir = os.path.join(GLM_OUT_DIR, subject)

            train_val_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                TRAINING_MODE)
            preloaded_betas = train_val_dataset.preload()

            idx = list(range(len(train_val_dataset)))
            kf = KFold(n_splits=NUM_CV_SPLITS, shuffle=True, random_state=1)

            test_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                           DECODER_TESTING_MODE,
                                           fmri_betas_transform=train_val_dataset.fmri_betas_transform,
                                           nn_latent_transform=train_val_dataset.nn_latent_transform)
            test_dataset.preload()

            # imagery_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, latent_vectors_file, f'imagery',  transform=bold_transform, latent_transform=latent_transform)
            # imagery_dataset.preload()

            results_dir = os.path.join(GLM_OUT_DIR,
                                       f'regression_results_mni_mmda_cv_shuffle_{TRAINING_MODE}/{subject}/{model_name}')

            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0, shuffle=False)

            HPs = [
                # HyperParameters(optimizer='SGD', lr=0.0001, wd=0.00, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.1, wd=0.00, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.01, wd=0.00, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.001, wd=0.00, dropout=False, loss='MSE'),

                # HyperParameters(optimizer='SGD', lr=0.01, wd=0.1, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.01, wd=1, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.01, wd=10, dropout=False, loss='MSE'),

                # HyperParameters(optimizer='ADAM', lr=0.001, wd=0, dropout=False, loss='MSE'),
                HyperParameters(optimizer='ADAM', lr=0.0001, wd=0, dropout=False, loss='MSE'),

                HyperParameters(optimizer='ADAM', lr=0.0001, wd=0.1, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='ADAM', lr=0.001, wd=0.1, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='ADAM', lr=0.01, wd=0.1, dropout=False, loss='MSE'),

                HyperParameters(optimizer='ADAM', lr=0.0001, wd=1, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='ADAM', lr=0.001, wd=1, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='ADAM', lr=0.01, wd=1, dropout=False, loss='MSE'),

                # HyperParameters(optimizer='ADAM', lr=0.0001, wd=10, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='ADAM', lr=0.001, wd=10, dropout=False, loss='MSE'),
            ]

            best_hp_setting = None
            best_hp_setting_val_loss = math.inf
            best_hp_setting_num_samples = None
            for hp in HPs:
                optim_type, lr, wd, dropout, loss_type = hp
                hp_str = hp.get_hp_string()
                print(hp_str)

                loss_fn = nn.MSELoss() if loss_type == 'MSE' else CosineDistance()
                imagery_loss_fn = CosineDistance()

                val_losses_for_folds = []
                num_samples_for_folds = []

                start = time.time()

                for fold, (train_idx, val_idx) in enumerate(kf.split(idx)):
                    gc.collect()
                    loss_fn = nn.MSELoss() if loss_type == 'MSE' else CosineDistance()
                    run_str = hp_str + f"fold_{fold}"

                    results_file_dir = f'{results_dir}/{run_str}'
                    checkpoint_dir = f'{results_dir}/networks/{run_str}'

                    os.makedirs(results_file_dir, exist_ok=True)
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    train_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                    TRAINING_MODE, subset=train_idx, fold=fold,
                                                    preloaded_betas=preloaded_betas)

                    val_dataset = COCOBOLDDataset(TWO_STAGE_GLM_DATA_DIR, subject, model_name, std_mean_dir,
                                                  TRAINING_MODE, subset=val_idx, fold=fold,
                                                  preloaded_betas=preloaded_betas,
                                                  fmri_betas_transform=train_dataset.fmri_betas_transform,
                                                  nn_latent_transform=train_dataset.nn_latent_transform
                                                  )
                    print(f"Train set size: {len(train_dataset)} | val set size: {len(val_dataset)}")

                    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True,
                                              drop_last=True)
                    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

                    model = LinearNet(train_dataset.bold_dim_size, train_dataset.latent_dim_size, dropout=dropout)
                    model = model.to(device)

                    sumwriter = SummaryWriter(f'{results_dir}/tensorboard/{run_str}', filename_suffix=f'')

                    optimizer = create_optimizer(optim_type)

                    epochs_no_improved_loss = 0
                    best_val_loss = math.inf
                    best_val_loss_num_samples = 0

                    num_samples_train_run = 0
                    for epoch in trange(MAX_EPOCHS, desc=f'training decoder for fold {fold}'):

                        train_loss, num_epoch_samples = train_decoder_epoch(model, train_loader, optimizer, loss_fn)

                        val_results = evaluate_decoder(model, val_loader, loss_fn)
                        num_samples_train_run += num_epoch_samples

                        sumwriter.add_scalar(f"Training/{loss_type} loss", train_loss, num_samples_train_run)
                        sumwriter.add_scalar(f"Val/{loss_type} loss", val_results['loss'], num_samples_train_run)

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

                        sumwriter.close()

                    # Final eval
                    model = LinearNet(train_dataset.bold_dim_size, train_dataset.latent_dim_size, dropout=dropout)
                    model = model.to(device)
                    model.load_state_dict(torch.load(f"{checkpoint_dir}/model_best_val.pt", map_location=device))

                    val_results = evaluate_decoder(model, val_loader, loss_fn, calc_eval_metrics=True)
                    pickle.dump(val_results, open(os.path.join(results_file_dir, "val_results.p"), 'wb'))

                    test_results = evaluate_decoder(model, test_loader, loss_fn, calc_eval_metrics=True)
                    pickle.dump(test_results, open(os.path.join(results_file_dir, "test_results.p"), 'wb'))

                    val_losses_for_folds.append(val_results['loss'])
                    num_samples_for_folds.append(best_val_loss_num_samples)
                    print(f"best val loss: {val_results['loss']:.4f}")
                    if len(val_losses_for_folds) == NUM_CV_SPLITS and np.mean(
                            val_losses_for_folds) < best_hp_setting_val_loss:
                        best_hp_setting_val_loss = np.mean(val_losses_for_folds)
                        best_hp_setting = hp
                        best_hp_setting_num_samples = int(np.mean(num_samples_for_folds))
                        print(
                            f"new best hp setting val loss: {np.mean(val_losses_for_folds):.4f} | num samples: {best_hp_setting_num_samples}")

                end = time.time()
                print(f"Elapsed time: {int(end - start)}s")

            # Re-train on full train set with best HP setting:
            optim_type, lr, wd, dropout, loss_type = best_hp_setting

            num_samples_per_epoch = (len(train_val_dataset) // BATCH_SIZE) * BATCH_SIZE
            best_hp_setting_num_epochs = (best_hp_setting_num_samples // num_samples_per_epoch) + 1
            print(
                f"Retraining {model_name} for {best_hp_setting_num_epochs} epochs on full train set with hp setting: ",
                best_hp_setting.get_hp_string())
            hp_str = best_hp_setting.get_hp_string() + "_full_train"
            model = LinearNet(train_val_dataset.bold_dim_size, train_val_dataset.latent_dim_size, dropout=dropout)
            model = model.to(device)

            full_train_loader = DataLoader(train_val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True,
                                           drop_last=True)
            loss_fn = nn.MSELoss() if loss_type == 'MSE' else CosineDistance()

            optimizer = create_optimizer(optim_type)

            sumwriter = SummaryWriter(f'{results_dir}/tensorboard/{hp_str}', filename_suffix=f'')
            checkpoint_dir = f'{results_dir}/networks/{hp_str}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            results_file_dir = f'{results_dir}/{hp_str}'
            os.makedirs(results_file_dir, exist_ok=True)

            num_samples_train_run = 0
            for epoch in trange(best_hp_setting_num_epochs, desc=f'training decoder on full train set'):
                train_loss, num_epoch_samples = train_decoder_epoch(model, full_train_loader, optimizer, loss_fn)

                num_samples_train_run += num_epoch_samples

                sumwriter.add_scalar(f"Training/{loss_type} loss", train_loss, num_samples_train_run)

                if num_samples_train_run >= best_hp_setting_num_samples:
                    print(f"reached {best_hp_setting_num_samples} samples. Terminating full train.")

                    torch.save(model.state_dict(), f"{checkpoint_dir}/model_best_val.pt")

                    test_results = evaluate_decoder(model, test_loader, loss_fn, calc_eval_metrics=True)

                    pickle.dump(test_results, open(os.path.join(results_file_dir, "test_results.p"), 'wb'))

                    best_dir = f'{results_dir}/best_hp/'
                    os.makedirs(best_dir, exist_ok=True)
                    pickle.dump(test_results, open(os.path.join(best_dir, "test_results.p"), 'wb'))

                    break
