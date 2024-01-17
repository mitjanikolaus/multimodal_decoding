import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import FEATURES_DIR, RESULTS_DIR

from glob import glob
import pickle

HP_KEYS = ["alpha", "model", "subject", "features", "training_mode", "testing_mode", "fold", "best_val_loss", "best_val_acc"]


def add_avg_subject(df):
    df_mean = df.copy()
    df_mean["subject"] = "average"
    df = pd.concat((df.copy(), df_mean))
    return df


distance_metric = 'cosine'
METRICS = [f'acc_{distance_metric}', f'acc_{distance_metric}_captions', f'acc_{distance_metric}_images']

def plot_metrics(data, hue_variable="model", row_variable=None, metrics=METRICS, ylim=(0.5, 1), add_avg_over_subjects=True, plot_legend=True):
    data_filtered = data[data.metric.isin(metrics)]
    if add_avg_over_subjects:
        data_filtered = add_avg_subject(data_filtered)

    sns.set_style("ticks", {'axes.grid' : True})
    g = sns.catplot(data_filtered, kind="bar", x="subject", y="value", row=row_variable, col="metric", col_order=metrics, height=5, aspect=1.5, hue=hue_variable)#, palette="Set3"
    g._legend.remove()
    bbox_extra_artists = None
    if plot_legend:
        lgd = g.fig.legend(loc='upper left', title="", bbox_to_anchor=(1, 0.9)) # ,ncol=2
        bbox_extra_artists = (lgd,)

    g.set(ylim=ylim)
    
    # plt.suptitle("Test Performance", fontsize=16)
    plt.tight_layout()
    
    plt.savefig(os.path.join(RESULTS_DIR, f"{hue_variable}_comparison_{metrics[0]}.png"), bbox_extra_artists=bbox_extra_artists, bbox_inches='tight', dpi=300)
    return g, data_filtered
    

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
