import nibabel as nib
import numpy as np
from glob import glob
import os
from os.path import join as opj
import pandas as pd
import shutil
import multiprocessing
from threading import Thread
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def extract_stim_ids_from_event_files(subject_folder):
    event_files = list(sorted(glob(opj(subject_folder, '**', '*events*.tsv'), recursive=True)))
    ids = {'images_train': [], 'images_test': [], 'captions_train': [], 'captions_test': []}
    for event_file in event_files:
        indices = []
        data = pd.read_csv(event_file, sep='\t')

        condition = np.array(data['condition_name'])
        trial_type = np.array(data['trial_type'])
        train_test = np.array(data['train_test'])
        oneback = np.array(data['one_back'])

        for img_cap, trn_tst, name in zip([1, 1, 2, 2], [1, 2, 1, 2],
                                          ['images_train', 'images_test', 'captions_train', 'captions_test']):
            mask = np.logical_and(trial_type == img_cap, train_test == trn_tst)
            stim = condition[mask]
            onb = oneback[mask]
            for idx, s in enumerate(stim):
                if s in ids[name]:
                    # print(f'Already there! {name}, ID: {s}, Oneback: {onb[idx]}')
                    pass
                else:
                    ids[name].append(s)
    return ids


def split_vision_and_language_beta_files_2phase_glm(beta_dir, wbf=False):
    r"""
    to make life easier, this function makes several subdirectories and creates symbolic links
    to the corresponding beta files. it also renames the links with the coco sample id.
    """
    mni_str = ''
    beta_file_addresses = list(
        sorted(glob(opj(beta_dir, f'unstructured{mni_str}', '**', 'beta_*.nii'), recursive=True)))
    subdirs = {
        'train_images': opj(beta_dir, f'betas_train_images{mni_str}'),
        'test_images': opj(beta_dir, f'betas_test_images{mni_str}'),
        'train_captions': opj(beta_dir, f'betas_train_captions{mni_str}'),
        'test_captions': opj(beta_dir, f'betas_test_captions{mni_str}'),
        'imagery': opj(beta_dir, f'betas_imagery{mni_str}'),
    }
    if wbf:
        subdirs['blank'] = opj(beta_dir, f'betas_blank{mni_str}')

    for _, diradd in subdirs.items():
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
            stim_id = int(beta_name[index + 11:endidx])
            slink_name = opj(subdirs['test_images'], f"beta_I{stim_id:06d}.nii")

        index = beta_name.find('train_image')
        if index != -1:
            if slink_name is not None:
                raise Exception('slink already defined')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index + 12:endidx])
            slink_name = opj(subdirs['train_images'], f"beta_I{stim_id:06d}.nii")

        index = beta_name.find('test_caption')
        if index != -1:
            if slink_name is not None:
                raise Exception('slink already defined')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index + 13:endidx])
            slink_name = opj(subdirs['test_captions'], f"beta_C{stim_id:06d}.nii")

        index = beta_name.find('train_caption')
        if index != -1:
            if slink_name is not None:
                raise Exception('slink already defined')
            endidx = beta_name.find('*bf(1)')
            stim_id = int(beta_name[index + 14:endidx])
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
    it uses multi-threading to boost the load speed. number of threads will be `threads_load` times total number of
    available cores
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
    beta_array = np.zeros((len(file_addresses),) + tuple(beta_shape), dtype=dtype)

    def load(indices, addresses, target_array, dtype):
        r"""
        loads beta files and puts them into target array according to the assigned indices
        """
        for idx in indices:
            target_array[idx, :, :, :] = nib.load(addresses[idx]).get_fdata().astype(dtype)

    threads = []
    for indices in indices_for_threads:
        t = Thread(target=load, args=(indices, file_addresses, beta_array, dtype))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if flatten:
        beta_array = beta_array.reshape(beta_array.shape[0], -1)
    return beta_array.squeeze()


def decoding_rank_score(predictions, originals, metric='cosine'):
    dist_mat = cdist(predictions, originals,
                     metric=metric)  # d(i,j) -> distance of the prediction of i to the original of j
    ranks = 1 - ((rankdata(dist_mat, axis=1) - 1) / (dist_mat.shape[1] - 1))
    ranks = ranks.diagonal()

    n = dist_mat.shape[0]
    scores = ranks.sum() / n
    return scores


if __name__ == "__main__":
    # extract_stim_ids_from_event_files('/home/leilar/Data/SEMREPS/SEMREPS_BIDS/sub-01')

    split_vision_and_language_beta_files_2phase_glm(
        '/mnt/HD1/milad/multimodal_decoding/glm_manual/two-stage-mni/sub-02', wbf=False)
    split_vision_and_language_beta_files_2phase_glm(
        '/mnt/HD1/milad/multimodal_decoding/glm_manual/two-stage-mni/sub-05', wbf=True)
