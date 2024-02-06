import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import FEATURES_DIR, RESULTS_DIR

from glob import glob
import pickle

HP_KEYS = ["alpha", "model", "subject", "features", "training_mode", "testing_mode", "mask", "best_val_loss", "best_val_acc", "best_val_mse", "ensemble", "num_voxels"]
METRIC_NAMES = {"acc_cosine": "pairwise_acc", "acc_cosine_captions": "pairwise_acc_captions", "acc_cosine_images": "pairwise_acc_images"}

def add_avg_subject(df):
    df_mean = df.copy()
    df_mean["subject"] = "average"
    return pd.concat((df.copy(), df_mean))


def load_results_data(distance_metrics = ["cosine"]):
    results_root_dir = os.path.expanduser(f'~/data/multimodal_decoding/glm/')
    
    metrics = ['val_loss', 'val_rsa', 'rsa', 'rsa_captions', 'rsa_images'] #, 'predictions', 'latents', 'stimulus_ids'
    for dist_metric in distance_metrics:
        metrics.extend([f'acc_{dist_metric}', f'acc_{dist_metric}_captions', f'acc_{dist_metric}_images', f'val_acc_{dist_metric}'])         
    for metric in ["spearmanr", "pearsonr"]:
        for matrix_metric in ["spearmanr", "pearsonr"]: 
            metrics.append(f"rsa_{metric}_{matrix_metric}")
        
    data = []
        
    result_files = sorted(glob(f"{results_root_dir}/*/*/*/results.p"))
    for result_file_path in result_files:        
        results = pickle.load(open(result_file_path, 'rb'))

        for metric in metrics:
            if metric in results.keys():
                data_item = {k: value for k, value in results.items() if k in HP_KEYS}
                data_item["metric"] = METRIC_NAMES[metric] if metric in METRIC_NAMES.keys() else metric
                data_item["value"] = results[metric]
                data.append(data_item)   

    df = pd.DataFrame.from_records(data)

    df["features"] = df.features.replace({"concat": "vision+lang"})
    df["model"] = df.model.replace({"mistral": "mistral-7b"})

    df["training_mode"] = df.training_mode.replace({"train": "modality-agnostic", "train_captions": "captions", "train_images": "images"})
    df["mask"] = df["mask"].fillna("whole_brain")

    df["model_feat"] = df.model + "_" + df.features

    df = df[df.ensemble != True]

    return df
