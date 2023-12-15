############################################
# Training multimodal linear brain decoders
# inputs can be of any modality
# outputs are uni-modal
############################################
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
import os
from glob import glob
import pickle
from torchvision.transforms import Compose
from decoding_utils import get_distance_matrix, to_tensor
from tqdm import trange, tqdm
import gc
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils import IMAGERY_SCENES, MODEL_FEATURES_FILES

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VAL_SPLIT_SIZE = 0.2
PATIENCE = 5


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

    data_modes = ['train', 'test', 'imagery']

    def __init__(self, bold_root_dir, subject, latent_vectors, mode='train', cache=True,
                 bold_transform=None, label_transform=None, latent_transform=None, blank_correction=True):
        """
        Args:
            bold_root_dir (str): BOLD root directory (parent of subjects' directories).
            subject (str): Subject ID.
            latent_vectors (dict): Dictionary of latent vectors for each stimulus. Keys are coco stim. ids.
            mode (str): 'train', 'test', or 'imagery'. You can append _images or _captions to make it unimodal
            cache (boolean): if `True`, data will be kept in the memory to optimize running time.
            bold_transform (callable, optional): Optional transformation to be applied
                on a sample.
            label_transform (callable, optional): Optional transformation to be applied
                on a sample label (stimulus id).
            latent_transform (callable, optional): optional transformation to be applied
                on latent vectors.
            blank_correction (boolean): If `True`, the blank image will be subtracted from the imagery patterns (if exists)
        """
        self.root_dir = os.path.join(bold_root_dir, subject)
        self.subject = subject
        self.imagery_scenes = IMAGERY_SCENES[subject]
        self.mode = mode
        self.bold_transform = bold_transform
        self.label_transform = label_transform
        self.latent_transform = latent_transform
        self.cache = cache
        self.blank = None
        self.blank_correction = blank_correction
        self.feature_key = ""
        for _, vec in latent_vectors.items():
            for key in vec:
                if 'feature' in key:
                    self.feature_key = key
                    print(key)
                    break
            break
        if self.feature_key == "":
            raise Exception('no feature found!')

        self.addresses = list(sorted(glob(os.path.join(self.root_dir, f'betas_{self.mode}*', '*.nii'))))
        self.stim_ids = []
        self.stim_types = []
        self.latent_vectors = []

        for addr in self.addresses:
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
            self.latent_vectors.append(latent_vectors[stim_id][self.feature_key])

        self.latent_vectors = np.array(self.latent_vectors)
        self.data = [None for _ in range(len(self.addresses))]

        brain_mask_address = os.path.join(self.root_dir, f'unstructured', 'mask.nii')
        self.brain_mask = nib.load(brain_mask_address).get_fdata().reshape(-1)
        self.brain_mask = np.logical_and(np.logical_not(np.isnan(self.brain_mask)), self.brain_mask != 0)
        self.bold_dim_size = self.brain_mask.sum()
        self.latent_dim_size = self.latent_vectors[0].shape[0]

        beta_blank_address = os.path.join(self.root_dir, f'betas_blank', 'beta_blank.nii')
        if os.path.exists(beta_blank_address):
            self.blank = nib.load(beta_blank_address).get_fdata().astype('float32').reshape(-1)
            self.blank = self.blank[self.brain_mask]

    def preload(self):
        for idx in trange(len(self.addresses)):
            sample = nib.load(self.addresses[idx]).get_fdata().astype('float32').reshape(-1)
            sample = sample[self.brain_mask]
            self.data[idx] = sample.copy()

    def get_latent_vector(self, idx):
        latent = self.latent_vectors[idx]
        if self.latent_transform is not None:
            latent = self.latent_transform(latent)
        return latent

    def get_brain_vector(self, idx):
        if self.cache and self.data[idx] is not None:
            sample = self.data[idx]
        else:
            sample = nib.load(self.addresses[idx]).get_fdata().astype('float32').reshape(-1)
            sample = sample[self.brain_mask]
            if self.cache:
                self.data[idx] = sample.copy()

        if self.mode == 'imagery' and self.blank is not None and self.blank_correction:
            sample = sample - self.blank
        if self.bold_transform is not None:
            sample = self.bold_transform(sample)
        return sample

    def get_stim_id(self, idx):
        sid = self.stim_ids[idx]
        if self.label_transform is not None:
            sid = self.label_transform(sid)
        return sid

    def get_stim_type(self, idx):
        return self.stim_types[idx]

    def __len__(self):
        return len(self.addresses)

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


def calc_mean_std_per_subject(fmri_data_dir, subject, latent_vectors_file, output_dir, training_mode,
                              model_name=None, overwrite=False):
    r"""
    Saves mean and std of training BOLD and/or latent vectors for a subject as dictionary (pickle) files.
    The resulting dictionary has two keys: (1): `"mean"` and (2): `"std"`, each linked to a numpy array representing
    the corresponding values.

    Args:
        `fmri_data_dir` (str): address to the BOLD root directory
        `subject` (str): Subject ID
        `latent_vectors_file` (str): address to the dictionaty containing latent vectors
        `output_dir` (str): address to the output directory
        `training_mode` (str): `"multimodal"`/`"captiononly"`/`"imageonly"`
        `model_std_mean_name` (str): name of the pickle file containing model mean and std. If `None`, model mean and std won't be computed.
        `overwrite` (boolean): if `True`, the mean and std will be recomputed and replaced by the old ones.
    """
    with open(latent_vectors_file, 'rb') as handle:
        latent_vectors = pickle.load(handle)

    calc_bold_std_mean = True
    model_std_mean_name = f'{model_name}_mean_std'

    m = training_mode
    if not overwrite:
        if os.path.exists(os.path.join(output_dir, f'bold_multimodal_mean_std_{m}.p')):
            calc_bold_std_mean = False
        if model_name is not None and os.path.exists(os.path.join(output_dir, f'{model_std_mean_name}_{m}.p')):
            model_std_mean_name = None

    dataset = COCOBOLDDataset(fmri_data_dir, subject, latent_vectors, m)

    os.makedirs(output_dir, exist_ok=True)
    if calc_bold_std_mean:
        print(f"Calculating Mean and STD of BOLD Signals for {m} Samples")

        bold_data_size = dataset.get_brain_vector(0).shape[0]
        bold_data = np.empty((len(dataset), bold_data_size))

        for idx in tqdm(range(len(dataset))):
            bd = dataset.get_brain_vector(idx)
            bold_data[idx] = bd

        mean_std = {'mean': bold_data.mean(axis=0).astype('float32'),
                    'std': bold_data.std(axis=0).astype('float32')}
        file_name = f'bold_multimodal_mean_std_{m}.p'
        pickle.dump(mean_std, open(os.path.join(output_dir, file_name), 'wb'), pickle.HIGHEST_PROTOCOL)

    if model_std_mean_name is not None:
        print(f"Calculating Mean and STD of Model Latent Variables for {m} Samples")

        model_data_size = dataset.get_latent_vector(0).shape[0]
        model_data = np.empty((len(dataset), model_data_size))

        for idx in tqdm(range(len(dataset))):
            md = dataset.get_latent_vector(idx)
            model_data[idx] = md

        mean_std = {'mean': model_data.mean(axis=0).astype('float32'),
                    'std': model_data.std(axis=0).astype('float32')}
        file_name = f'{model_std_mean_name}_{m}.p'
        pickle.dump(mean_std, open(os.path.join(output_dir, file_name), 'wb'), pickle.HIGHEST_PROTOCOL)


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


def train_decoder_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    cum_loss = 0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, latents, ids, types = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, latents.to(device))
        cum_loss += loss.item()
        loss.backward()
        optimizer.step()
    cum_loss /= len(train_loader)
    return cum_loss


def evaluate_decoder(net, test_loader, loss_fn, distance_metrics, device):
    r"""
    evaluates decoder on test bold signals
    returns the predicted vectors and loss values
    `distance_metrics` is a list of string containing distance metric names
    """
    net.eval()
    cum_loss = 0
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            test_inputs, test_latents, test_ids, test_types = data
            outputs = net(test_inputs.to(device))
            test_loss = loss_fn(outputs, test_latents.to(device))
            cum_loss += test_loss.item()
            predictions.append(outputs.cpu().numpy())
    cum_loss /= len(test_loader)
    predictions = np.concatenate(predictions, axis=0)
    predictions = (predictions-predictions.mean(axis=0))/predictions.std(axis=0)

    dist_matrices = {'classes': test_ids,
                     'types': test_types}  # order of classes, order of types (image, caption, imagery)
    for metric in distance_metrics:
        dist_matrices[metric] = get_distance_matrix(predictions, test_latents, metric)
    return predictions, cum_loss, dist_matrices


SUBJECTS = ['sub-01', 'sub-02', 'sub-04', 'sub-05', 'sub-07']  # TODO 'sub-03'

# model_names = ['GPT2XL_AVG', 'VITL16_ENCODER','RESNET152_AVGPOOL', 'GPT2XL_AVG_PCA768', 'VITL16_ENCODER_PCA768']
MODEL_NAMES = ['RESNET152_AVGPOOL']
# MODEL_NAMES = ['CLIP_L', 'CLIP_V', 'CLIP_L_PCA768', 'CLIP_V_PCA768', 'RESNET152_AVGPOOL']  # RESNET152_AVGPOOL_PCA768
# MODEL_NAMES = ['BERT_LARGE', 'CLIP_L', 'CLIP_V', 'VITL16_ENCODER', 'RESNET152_AVGPOOL', 'GPT2XL_AVG']
TRAINING_MODE = ['train', 'train_captions', 'train_images'][0]
DECODER_TESTING_MODE = ['test', 'test_captions', 'test_images'][0]

GLM_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/glm/")
FMRI_DATA_DIR = os.path.expanduser("~/data/multimodal_decoding/fmri/")
DISTANCE_METRICS = ['cosine', 'euclidean']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    print("device: ", device)
    os.makedirs(GLM_OUT_DIR, exist_ok=True)

    for subject in SUBJECTS:
        print(subject)
        for model_name in MODEL_NAMES:
            print(model_name)
            latent_vectors_file = MODEL_FEATURES_FILES[model_name]
            two_stage_glm_dir = os.path.join(FMRI_DATA_DIR, "glm_manual/two-stage-mni/")
            std_mean_dir = os.path.join(GLM_OUT_DIR, subject)

            # calculating dataset mean and std for the normalization
            calc_mean_std_per_subject(
                fmri_data_dir=two_stage_glm_dir,
                subject=subject,
                latent_vectors_file=latent_vectors_file,
                output_dir=std_mean_dir,
                training_mode=TRAINING_MODE,
                model_name=model_name
            )

            latent_vectors = pickle.load(open(latent_vectors_file, 'rb'))

            # bold transform
            with open(os.path.join(std_mean_dir, f'bold_multimodal_mean_std_{TRAINING_MODE}.p'), 'rb') as handle:
                bold_mean_std = pickle.load(handle)
            bold_transform = Compose([
                Normalize(bold_mean_std['mean'], bold_mean_std['std']),
                to_tensor
            ])

            # latent transform
            with open(os.path.join(std_mean_dir, f'{model_name}_mean_std_{TRAINING_MODE}.p'), 'rb') as handle:
                model_mean_std = pickle.load(handle)
            latent_transform = Compose([
                Normalize(model_mean_std['mean'], model_mean_std['std']),
                to_tensor
            ])

            # preloading datasets for faster execution
            print("preloading bold train dataset")
            train_val_dataset = COCOBOLDDataset(two_stage_glm_dir, subject, latent_vectors, f'{TRAINING_MODE}',
                                                bold_transform=bold_transform, latent_transform=latent_transform)
            train_val_dataset.preload()

            idx = list(range(len(train_val_dataset)))
            train_idx, val_idx = train_test_split(idx, test_size=VAL_SPLIT_SIZE, random_state=1)
            train_dataset = Subset(train_val_dataset, train_idx)
            val_dataset = Subset(train_val_dataset, val_idx)
            print(f"Train set size: {len(train_dataset)} | val set size: {len(val_dataset)}")

            print("preloading bold test dataset")
            test_dataset = COCOBOLDDataset(two_stage_glm_dir, subject, latent_vectors, f'{DECODER_TESTING_MODE}',
                                           bold_transform=bold_transform, latent_transform=latent_transform)
            test_dataset.preload()

            # imagery_dataset = COCOBOLDDataset(two_stage_glm_dir, subject, latent_vectors, f'imagery',  transform=bold_transform, latent_transform=latent_transform)
            # imagery_dataset.preload()

            results_dir = os.path.join(GLM_OUT_DIR,
                                       f'regression_results_mni_mmda_val_set_{TRAINING_MODE}/{subject}/{model_name}')

            batch_size = len(train_val_dataset) // 4
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
            # test_images_loader = DataLoader(test_images_dataset, batch_size=len(test_images_dataset), num_workers=0, shuffle=False)
            # test_captions_loader = DataLoader(test_captions_dataset, batch_size=len(test_captions_dataset), num_workers=0, shuffle=False)
            # imagery_loader = DataLoader(imagery_dataset, batch_size=len(imagery_dataset), num_workers=0, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0, shuffle=False)

            # training setting
            max_epochs = 400
            if TRAINING_MODE != 'train':
                max_epochs = max_epochs * 2
                batch_size = batch_size * 2
            print('batch size:', batch_size)
            HPs = [
                # HyperParameters(optimizer='SGD', lr=0.005, wd=0.00, dropout=False, loss='MSE'),
                HyperParameters(optimizer='SGD', lr=0.010, wd=0.00, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.050, wd=0.00, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.100, wd=0.00, dropout=False, loss='MSE'),

                # HyperParameters(optimizer='SGD', lr=0.005, wd=0.01, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.010, wd=0.01, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.050, wd=0.01, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.100, wd=0.01, dropout=False, loss='MSE'),

                # HyperParameters(optimizer='SGD', lr=0.005, wd=0.10, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.010, wd=0.10, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.050, wd=0.10, dropout=False, loss='MSE'),
                # HyperParameters(optimizer='SGD', lr=0.100, wd=0.10, dropout=False, loss='MSE'),
            ]

            for hp in HPs:
                optim_type, lr, wd, dropout, loss_type = hp
                hp_str = hp.get_hp_string()
                print(hp_str)

                distance_dir = f'{results_dir}/distance_matrix/{hp_str}'
                os.makedirs(distance_dir, exist_ok=True)

                net = LinearNet(train_val_dataset.bold_dim_size, train_val_dataset.latent_dim_size, dropout=dropout)
                if torch.cuda.is_available():
                    net = net.cuda()

                sumwriter = SummaryWriter(f'{results_dir}/tensorboard/{hp_str}', filename_suffix=f'')
                checkpoint_dir = f'{results_dir}/networks/{hp_str}'
                os.makedirs(checkpoint_dir, exist_ok=True)

                best_net_states = {}
                gc.collect()
                loss_fn = nn.MSELoss() if loss_type == 'MSE' else CosineDistance()
                imagery_loss_fn = CosineDistance()

                if optim_type == 'SGD':
                    optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=lr, weight_decay=wd)
                elif optim_type == 'ADAM':
                    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
                else:
                    raise RuntimeError("Unknown optimizer: ", optim_type)

                epochs_no_improved_loss = 0
                start = time.time()
                for epoch in trange(max_epochs, desc=f'training decoder'):

                    train_loss = train_decoder_epoch(net, train_loader, optimizer, loss_fn, device=device)

                    val_predictions, val_loss, _ = evaluate_decoder(net, val_loader, loss_fn,
                                                                    distance_metrics=[],
                                                                    device=device)
                    test_predictions, test_loss, distance_matrices = evaluate_decoder(net, test_loader, loss_fn,
                                                                                      distance_metrics=DISTANCE_METRICS,
                                                                                      device=device)

                    # imagery_predictions, imagery_loss = evaluate_decoder(net, imagery_loader, imagery_loss_fn, False, device=DEVICE)

                    # test_loss = (testing_images_loss+testing_captions_loss)/2

                    sumwriter.add_scalar(f"Training/{loss_type} loss", train_loss, epoch)
                    sumwriter.add_scalar(f"Val/{loss_type} loss", val_loss, epoch)
                    sumwriter.add_scalar(f"Testing/{loss_type} loss", test_loss, epoch)

                    # best decoder
                    key = f'best_val'
                    if key not in best_net_states:
                        best_net_states[key] = {'net': net.state_dict(), 'epoch': epoch, 'value': val_loss}
                    else:
                        best_val_loss = best_net_states[key]['value']
                        if val_loss < best_val_loss:
                            epochs_no_improved_loss = 0
                            best_net_states[key] = {'net': net.state_dict(), 'epoch': epoch, 'value': val_loss}

                            torch.save(best_net_states[key]['net'], f"{checkpoint_dir}/net_{key}")

                            with open(os.path.join(distance_dir, "distance_matrix.p"), 'wb') as handle:
                                pickle.dump(distance_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)
                        else:
                            epochs_no_improved_loss += 1

                    if epochs_no_improved_loss >= PATIENCE:
                        print(f"Loss did not improve for {epochs_no_improved_loss} epochs. Terminating training.")
                        break

                    # # best imagery decoder (based on cosine loss)
                    # key = f'best_imagery'
                    # if key not in best_net_states:
                    #     best_net_states[key] = {'net':net.state_dict(), 'epoch':epoch, 'value':imagery_loss}
                    # else:
                    #     v = best_net_states[key]['value']
                    #     if imagery_loss <= v:
                    #         best_net_states[key] = {'net':net.state_dict(), 'epoch':epoch, 'value':imagery_loss}

                end = time.time()
                print(f"Elapsed time: {end - start}s")
                sumwriter.close()
