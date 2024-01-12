import nibabel as nib
import numpy as np
from glob import glob
import os
from os.path import join as opj

import torch
from tqdm import tqdm
import shutil
import multiprocessing
from threading import Thread
from datetime import datetime
import warnings
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
import pandas as pd


class HyperParameters:
    def __init__(self, optim_type='SGD', lr=0.01, wd=0.01, dropout=False, loss='MSE', alpha=None, full_train=False):
        self.optim_type = optim_type
        self.lr = lr
        self.wd = wd
        self.dropout = dropout
        self.loss_type = loss
        self.alpha = alpha
        self.full_train = full_train

    def to_string(self):
        if self.alpha is not None:
            descr = (f'[alpha:{float(self.alpha)}]'
                     f'[loss:{float(self.loss_type)}]')
        else:
            descr = (f"[optim:{self.optim_type}]"
                    f"[lr:{str(self.lr).replace('.', '-')}]"
                    f"[wd:{str(self.wd).replace('.', '-')}]"
                    f"[drop:{self.dropout}]"
                    f"[loss:{self.loss_type}]")
        if self.full_train:
            descr += "_full_train"
        return descr


class DecodingContainer():
    def __init__(self):
        self.results = {}
        self.definitions = {
            'predicted_z_vvvv': 'predicted vision latent vectors from visual trials using vision-vision mapping',
            'predicted_z_llll': 'predicted language latent vectors from lingual trials using language-language mapping',
            'predicted_z_lvvv': 'predicted vision latent vectors from lingual trials using vision-vision mapping',
            'predicted_z_vlll': 'predicted language latent vectors from visual trials using language-language mapping',

            'score_vvvv': 'decoding accuracy for predicting vision latent vectors from visual trials using vision-vision mapping',
            'score_llll': 'decoding accuracy for predicting language latent vectors from lingual trials using language-language mapping',
            'score_lvvv': 'decoding accuracy for predicting vision latent vectors from lingual trials using vision-vision mapping',
            'score_vlll': 'decoding accuracy for predicting language latent vectors from visual trials using language-language mapping',
            
            'predicted_z_ivvv': 'predicted vision latent vectors from imagery trials using vision-vision mapping',
            'predicted_z_illl': 'predicted language latent vectors from imagery trials using language-language mapping',
            
            'voxel_count': 'number of voxels included in the decoding',
            'mask_name': 'name of the mask used for the decoding',
            'task_name': 'decoding task id (name)'
        }
    
    def __getattr__(self, key):
        if key in self.results:
            return self.results[key]
        return super().__getattr__(key)


def extract_stim_ids_from_event_files(subject_folder):
    event_files = list(sorted(glob(opj(subject_folder,'**','*events*.tsv'), recursive=True)))
    ids = {'images_train':[], 'images_test':[], 'captions_train':[], 'captions_test':[]}
    for event_file in event_files:
        indices = []
        data = pd.read_csv(event_file, sep='\t')
        
        condition = np.array(data['condition_name'])
        trial_type = np.array(data['trial_type'])
        train_test = np.array(data['train_test'])
        oneback = np.array(data['one_back'])

        for img_cap, trn_tst, name in zip([1,1,2,2],[1,2,1,2],['images_train', 'images_test', 'captions_train', 'captions_test']):
            mask = np.logical_and(trial_type == img_cap, train_test == trn_tst)
            stim = condition[mask]
            onb  = oneback[mask]
            for idx, s in enumerate(stim):
                if s in ids[name]:
                    # print(f'Already there! {name}, ID: {s}, Oneback: {onb[idx]}')
                    pass
                else:
                    ids[name].append(s)
    return ids


def split_vision_and_language_beta_files(beta_dir, wbf=False):
    r"""
    to make life easier, this function makes two subdirectories (images/captions) and creates symbolic links
    to the corresponding beta files (latent variables) plus the bias beta. it also rename the links with the latent variable id.
    bias betas be the files corresponding to train_image or train_caption.
    """
    beta_file_addresses = list(sorted(glob(opj(beta_dir, 'beta_*.nii'))))
    subdirs = {
        'train_images'  : opj(beta_dir,'betas_train_images'),
        'test_images'   : opj(beta_dir,'betas_test_images'),
        'train_captions': opj(beta_dir,'betas_train_captions'),
        'test_captions' : opj(beta_dir,'betas_test_captions'),
        'imagery'       : opj(beta_dir,'betas_imagery'),
        }
    if wbf:
        subdirs['blank'] = opj(beta_dir,'betas_blank')

    for _,diradd in subdirs.items():
        if os.path.exists(diradd):
            shutil.rmtree(diradd)
        os.mkdir(diradd)

    for add in beta_file_addresses:
        beta_file = nib.load(add)
        beta_name = str(beta_file.header['descrip'].astype(str))
        
        slink_name = None
        
        index = beta_name.find('blank')
        if index != -1:
            endidx = beta_name.find('*bf(1)')
            slink_name = opj(subdirs['blank'], f"beta_blank.nii")

        index = beta_name.find('imagery_')
        if index != -1:
            endidx = beta_name.find('*bf(1)')
            slink_name = opj(subdirs['imagery'], f"beta_{beta_name[index:endidx]}.nii")

        index = beta_name.find('test_image')
        if index != -1:
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index+11:endidx])
            slink_name = opj(subdirs['test_images'], f"beta_I{stim_id:06d}.nii")

        index = beta_name.find('train_image*bf(1)')
        if index != -1:
            slink_name = opj(subdirs['train_images'], f"beta_VZbias.nii")

        index = beta_name.find('VZ')
        if index != -1:
            slink_name = opj(subdirs['train_images'], f"beta_{beta_name[index:index+7]}.nii")

        index = beta_name.find('test_caption')
        if index != -1:
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index+13:endidx])
            slink_name = opj(subdirs['test_captions'], f"beta_C{stim_id:06d}.nii")
        
        index = beta_name.find('train_caption*bf(1)')
        if index != -1:
            slink_name = opj(subdirs['train_captions'], f"beta_LZbias.nii")

        index = beta_name.find('LZ')
        if index != -1:
            slink_name = opj(subdirs['train_captions'], f"beta_{beta_name[index:index+7]}.nii")
        
        if slink_name:
            print(slink_name)
            os.symlink(add, slink_name)


def split_vision_and_language_beta_files_2phase_glm(beta_dir, wbf=False):
    r"""
    to make life easier, this function makes several subdirectories and creates symbolic links
    to the corresponding beta files. it also renames the links with the coco sample id.
    """
    mni_str = ''
    beta_file_addresses = list(sorted(glob(opj(beta_dir, f'unstructured{mni_str}', '**', 'beta_*.nii'), recursive=True)))
    subdirs = {
        'train_images'  : opj(beta_dir,f'betas_train_images{mni_str}'),
        'test_images'   : opj(beta_dir,f'betas_test_images{mni_str}'),
        'train_captions': opj(beta_dir,f'betas_train_captions{mni_str}'),
        'test_captions' : opj(beta_dir,f'betas_test_captions{mni_str}'),
        'imagery'       : opj(beta_dir,f'betas_imagery{mni_str}'),
        }
    if wbf:
        subdirs['blank'] = opj(beta_dir,f'betas_blank{mni_str}')

    for _,diradd in subdirs.items():
        if os.path.exists(diradd):
            shutil.rmtree(diradd)
        os.mkdir(diradd)

    all_slink_names = set()
    for add in beta_file_addresses:
        beta_file = nib.load(add)
        beta_name = str(beta_file.header['descrip'].astype(str))
        
        slink_name = None
        
        index = beta_name.find('blank')
        if index != -1:
            endidx = beta_name.find('*bf(1)')
            slink_name = opj(subdirs['blank'], f"beta_blank.nii")

        index = beta_name.find('imagery_')
        if index != -1:
            if slink_name is not None:
                raise Exception('slink already defined')
            endidx = beta_name.find('*bf(1)')
            slink_name = opj(subdirs['imagery'], f"beta_{beta_name[index:endidx]}.nii")

        index = beta_name.find('test_image')
        if index != -1:
            if slink_name is not None:
                raise Exception('slink already defined')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index+11:endidx])
            slink_name = opj(subdirs['test_images'], f"beta_I{stim_id:06d}.nii")

        index = beta_name.find('train_image')
        if index != -1:
            if slink_name is not None:
                raise Exception('slink already defined')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index+12:endidx])
            slink_name = opj(subdirs['train_images'], f"beta_I{stim_id:06d}.nii")

        index = beta_name.find('test_caption')
        if index != -1:
            if slink_name is not None:
                raise Exception('slink already defined')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index+13:endidx])
            slink_name = opj(subdirs['test_captions'], f"beta_C{stim_id:06d}.nii")
        
        index = beta_name.find('train_caption')
        if index != -1:
            if slink_name is not None:
                raise Exception('slink already defined')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index+14:endidx])
            slink_name = opj(subdirs['train_captions'], f"beta_C{stim_id:06d}.nii")
        
        if slink_name:
            if slink_name in all_slink_names:
                raise Exception('symlink already added!')
            all_slink_names.add(slink_name)
            print(slink_name)
            os.symlink(add, slink_name)


def load_beta_files(file_addresses, dtype=np.float64, threads_load=0.8, flatten=True):
    r"""
    loads beta files from the disk (`file_addresses`) and returns them as a 4D numpy array.
    it uses multi-threading to boost the load speed. number of threads will be `threads_load` times total number of available cores
    """
    if isinstance(file_addresses, str):
        file_addresses = [file_addresses]

    nb_cores = multiprocessing.cpu_count()
    nb_thread = np.ceil(nb_cores * threads_load)
    nb_thread = int(min(nb_thread, len(file_addresses)))

    print(nb_thread)
    indices_for_threads = np.arange(len(file_addresses))
    indices_for_threads = np.array_split(indices_for_threads, nb_thread)

    beta_shape = nib.load(file_addresses[0]).shape
    beta_array = np.zeros((len(file_addresses),)+tuple(beta_shape), dtype=dtype)

    def load(indices, addresses, target_array, dtype):
        r"""
        loads beta files and puts them into target array according to the assigned indices
        """
        for idx in indices:
            target_array[idx,:,:,:] = nib.load(addresses[idx]).get_fdata().astype(dtype)

    threads = []
    for indices in indices_for_threads:
        t = Thread(target=load, args=(indices, file_addresses, beta_array, dtype))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if flatten:
        beta_array = beta_array.reshape(beta_array.shape[0],-1)
    return beta_array.squeeze()


def load_beta_files_sequential(file_addresses, stack=True, dtype=np.float64):
    r"""
    loads beta files from the disk (`file_addresses`) and returns them.
    if `stack` is `True`, the result will be a single numpy array (4d) that includes all beta values.
    otherwise, a list of numpy arrays (3d) will be returned.
    """
    if isinstance(file_addresses, str):
        file_addresses = [file_addresses]
    beta_files = []
    for add in tqdm(file_addresses):
        beta_files.append(nib.load(add).get_fdata().astype(dtype))
    if len(beta_files) == 1:
        return beta_files[0]
    if stack:
        return np.array(beta_files)
    return beta_files


def compute_inverse_covariance(w):
    r"""
    computes (ww^t)^-1 using pinv.
    """
    cov = np.matmul(w, w.T)
    icov = np.linalg.pinv(cov)
    rank_cov = np.linalg.matrix_rank(cov)
    if rank_cov < icov.shape[0]:
        warnings.warn(f'RANK DEFICIENT Covariance Matrix!!! rank is {rank_cov} instead of {icov.shape[0]}', stacklevel=2)
    return icov


def decode(y, w, invcov=None):
    r"""
    y (test brain signals): np.array of size (nb_stimulus, nb_voxel)
    w (weight matrix):      np.array of size (nb_latent,   nb_voxel)
    """
    a = np.matmul(y, w.T)
    if invcov is None:
        invcov = compute_inverse_covariance(w)
    x = np.matmul(a, invcov)
    return x


def decoding_rank_score(predictions, originals, metric='cosine'):
    dist_mat = cdist(predictions, originals, metric=metric)              # d(i,j) -> distance of the prediction of i to the original of j
    ranks    = 1 - ((rankdata(dist_mat,axis=1)-1)/(dist_mat.shape[1]-1)) 
    ranks    = ranks.diagonal()
    
    n = dist_mat.shape[0]
    scores = ranks.sum() / n
    return scores


def get_distance_matrix(predictions, originals, metric='cosine'):
    return cdist(predictions, originals, metric=metric)


def get_nearest_neighbors_indices(vector, dataset_vectors, n_neighbors, mean_correction=None, metric='cosine'):
    r"""
    extracts neighboring vectors of each given vector with respect to the dataset.
    metric defines the distance metric for finding neighbors
    """
    # if vector is 2-dim, the neighbors will be returned for each sample
    if np.ndim(vector) == 1:
        vector = vector[np.newaxis,:]

    if mean_correction is not None:
        if np.ndim(mean_correction) == 1:
            mean_correction = mean_correction[np.newaxis,:]
        vector = vector.copy() - mean_correction
        dataset_vectors = dataset_vectors.copy() - mean_correction
    
    dists = cdist(vector, dataset_vectors, metric=metric)
    nearests_ids   = np.argsort(dists, axis=1)[:, :n_neighbors]
    nearests_dists = np.sort(dists, axis=1)[:, :n_neighbors]

    return nearests_ids, nearests_dists


def generate_random_mask(brain_mask, top_percentile=20):
    if isinstance(top_percentile, float):
        top_percentile = [top_percentile]
    
    valid_cnt     = np.count_nonzero(brain_mask)
    selection_cnt = [(int)(np.ceil(p*valid_cnt/100)) for p in top_percentile]
    valid_indices = np.argwhere(brain_mask)
    masks = []
    for cnt in selection_cnt:
        m = np.full_like(brain_mask, False, dtype=bool)
        indices = valid_indices.copy()
        np.random.shuffle(indices)
        m[indices[:cnt]]=True
        masks.append(m)
    if len(masks) == 1:       # anything with NaN will be False
        return masks[0]
    return masks


def denormalize(vectors, mean, std, normalize_first=False, normalize_axis=None):
    r"""
    if `normalize_first` is `True`, the vectors will be first normalized with their own mean and std (along the `normalize_axis`)
    """
    # print(vectors.mean(axis=normalize_axis, keepdims=True), vectors.std(axis=normalize_axis, keepdims=True))
    if normalize_first:
        vmean = vectors.mean(axis=normalize_axis, keepdims=True)
        vstd  = vectors.std(axis=normalize_axis, keepdims=True)
        vectors = vectors - vmean
        vectors = vectors / vstd

    # vectors = vectors / 2
    vectors = vectors * std
    vectors = vectors + mean

    return vectors


def to_tensor(v):
    return torch.from_numpy(v)


if __name__ == "__main__":
    # extract_stim_ids_from_event_files('/home/leilar/Data/SEMREPS/SEMREPS_BIDS/sub-01')
    # split_vision_and_language_beta_files('/home/milad/projects/multimodal_decoding/glm_manual/clip_vit_b16_crop/sub-01/betas')
    # split_vision_and_language_beta_files('/home/milad/projects/multimodal_decoding/glm_manual/clip_vit_b16_crop_8sess_01_02_03_05_06_07_09_11/sub-01/betas')
    # split_vision_and_language_beta_files('/home/milad/projects/multimodal_decoding/glm_manual/clip_vit_b16_crop_5sess_01_03_05_07_09/sub-01/betas')
    # split_vision_and_language_beta_files('/home/milad/projects/multimodal_decoding/glm_manual/clip_vit_l14_336px_crop_zscore/sub-02/betas_fs6_mni', wbf=False)
    # split_vision_and_language_beta_files('/home/milad/projects/multimodal_decoding/glm_manual/clip_vit_l14_336px_crop_zscore/sub-03/betas_fs6_mni', wbf=True)
    # split_vision_and_language_beta_files('/home/milad/projects/multimodal_decoding/glm_manual/clip_vit_l14_336px_crop_zscore/sub-04/betas_fs6_mni', wbf=True)
    # split_vision_and_language_beta_files('/home/milad/projects/multimodal_decoding/glm_manual/clip_vit_l14_336px_crop_zscore/sub-07/betas_mni', wbf=True)
    # split_vision_and_language_beta_files('/home/milad/projects/multimodal_decoding/glm_manual/resnet152_avgpool_gpt2xl_avg_pca_768_crop_zscore/sub-01/betas_mni', wbf=False)

    # split_vision_and_language_beta_files_2phase_glm('/home/milad/projects/multimodal_decoding/glm_manual/two-stage/sub-01')
    split_vision_and_language_beta_files_2phase_glm('/mnt/HD1/milad/multimodal_decoding/glm_manual/two-stage-mni/sub-02', wbf=False)
    split_vision_and_language_beta_files_2phase_glm('/mnt/HD1/milad/multimodal_decoding/glm_manual/two-stage-mni/sub-05', wbf=True)
    # split_vision_and_language_beta_files_2phase_glm('/mnt/HD1/milad/multimodal_decoding/glm_manual/two-stage-mni/sub-07', wbf=True, mni=False)

    # beta_addresses = list(sorted(glob('/home/milad/projects/multimodal_decoding/glm_manual/glm_clip_vit_b16/sub-01/betas/betas_train_images/*.nii')))
    # mask_address = list(sorted(glob('/home/milad/projects/multimodal_decoding/glm_manual/glm_clip_vit_b16/sub-01/betas/*mask*.nii')))

    # mask  = load_beta_files(mask_address, dtype=np.int16)
    # start = datetime.now()
    # betas = load_beta_files(beta_addresses, dtype=np.float32)
    # betas_seq = load_beta_files(beta_addresses, dtype=np.float32)
    # print(datetime.now() - start)
    # betas_seq = load_beta_files_sequential(image_beta_addresses[:256], dtype=np.float32)
    # print(beta.shape, beta.dtype, betas.shape, betas.dtype)

    # betas_no_nans = betas.reshape((betas.shape[0],-1))[:, mask.reshape(-1) != 0]
    # betas_seq_no_nans = betas_seq.reshape((betas_seq.shape[0],-1))[:, mask.reshape(-1) != 0]
    
    # print(np.all(betas_no_nans.reshape(-1) == betas_seq_no_nans.reshape(-1)))

    # a = np.array([
    #     [4, 0, 2, 10],
    #     [1, 2, 3, 4],
    #     [10, 10, 10, 1],
    #     [0, 5, 5, 0]
    # ])

    # diag = a.diagonal()[np.newaxis, :]

    # comp = diag < a

    # print(a)
    # print()
    # print(diag)
    # print()
    # # print(comp)
    # # print()
    # print(1 - ((rankdata(a,axis=0)-1)/(a.shape[0]-1)))
    # # print()
    # # print(rankdata(a,axis=1)/a.shape[0])

    # import pickle
    # with open('/home/milad/projects/multimodal_decoding/latent_vectors/clip/ViT-B16_selected_coco_dataset_crop_vl.pickle', 'rb') as f:
    #     dataset = pickle.load(f)
    
    # stim_ids = list(dataset.keys())
    # dataset_vectors = np.array([dataset[k]['visual_feature'] for k in stim_ids])
    # vectors = dataset_vectors[:2,:]

    # nn_indices = get_nearest_neighbors_indices(vectors, dataset_vectors, 5)
    
    
    # a = np.random.randn(3,10)
    # b = np.random.randn(3,10)
    # print(np.diag(1 - cdist(a.T,b.T,metric='cosine')))
    # print(select_voxels_split_run_correlation([a,b], 10))
    # print(select_voxels_split_run_correlation([a,b], 20))
    # print(select_voxels_split_run_correlation([a,b], 30))
    # print(select_voxels_split_run_correlation([a,b], 40))
    # print(select_voxels_split_run_correlation([a,b], 50))
    # print(select_voxels_split_run_correlation([a,b], 60))
    # print(select_voxels_split_run_correlation([a,b], 70))
    # print(select_voxels_split_run_correlation([a,b], 80))
    # print(select_voxels_split_run_correlation([a,b], 90))
    # print(select_voxels_split_run_correlation([a,b], 100))

    # beta_imagery_mask_add = f'/home/milad/projects/multimodal_decoding/glm_manual/clip_vit_b16_crop/sub-01/betas/mask.nii'
    # brain_mask = load_beta_files(beta_imagery_mask_add, dtype=np.int16) != 0
    # a = generate_random_mask(brain_mask, [0.01, 0.5, 0.2])
    pass
