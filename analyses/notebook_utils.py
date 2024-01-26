import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import FEATURES_DIR, RESULTS_DIR

from glob import glob
import pickle

HP_KEYS = ["alpha", "model", "subject", "features", "training_mode", "testing_mode", "fold", "best_val_loss", "best_val_acc", "best_val_mse", "ensemble"]
METRIC_NAMES = {"acc_cosine": "pairwise_acc", "acc_cosine_captions": "pairwise_acc_captions", "acc_cosine_images": "pairwise_acc_images"}
COLORS_PLOT_CATEGORICAL = [
    "#000000",
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    # "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
]

def add_avg_subject(df):
    df_mean = df.copy()
    df_mean["subject"] = "average"
    return pd.concat((df.copy(), df_mean))


def load_results_data(distance_metrics = ["cosine"]):
    results_root_dir = os.path.expanduser(f'~/data/multimodal_decoding/glm/')
    
    metrics = ['val_loss', 'val_rsa', 'rsa'] #, 'predictions', 'latents', 'stimulus_ids'
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

    df["model_feat"] = df.model + "_" + df.features

    df = df[df.ensemble != True]

    return df
