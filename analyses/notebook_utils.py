import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import FEATURES_DIR

from glob import glob
import pickle
from tqdm import tqdm

HP_KEYS = ["alpha", "model", "subject", "features", "training_mode", "testing_mode", "fold", "best_val_loss", "best_val_acc"]

def load_results_data(distance_metrics = ["cosine"]):
    results_root_dir = os.path.expanduser(f'~/data/multimodal_decoding/glm/')
    
    metrics = ['val_loss', 'val_rsa', 'rsa'] #, 'predictions', 'latents', 'stimulus_ids'
    for dist_metric in distance_metrics:
        metrics.extend([f'acc_{dist_metric}', f'acc_{dist_metric}_captions', f'acc_{dist_metric}_images', f'val_acc_{dist_metric}'])         
    
    data = []
        
    result_files = sorted(glob(f"{results_root_dir}/*/*/*/results.p"))
    for result_file_path in result_files:        
        results = pickle.load(open(result_file_path, 'rb'))

        for metric in metrics:
            if metric in results.keys():
                data_item = {k: value for k, value in results.items() if k in HP_KEYS}
                data_item["metric"] = metric
                data_item["value"] = results[metric]
                data.append(data_item)   

    df = pd.DataFrame.from_records(data)

    df["model"] = df.model + "_" + df.features

    return df
