import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import FEATURES_DIR, RESULTS_DIR, SUBJECTS, NUM_TEST_STIMULI

from glob import glob
import pickle

HP_KEYS = ["alpha", "model", "subject", "features", "vision_features", "training_mode", "mask", "num_voxels"]
METRIC_NAMES = {"acc_cosine": "pairwise_acc", "acc_cosine_captions": "pairwise_acc_captions", "acc_cosine_images": "pairwise_acc_images"}


def plot_metric(data, kind="bar", x_variable="model_feat", order=None, hue_variable="model_feat", hue_order=None, metric="pairwise_acc_mean", ylim=(0.5, 1), plot_legend=True, palette=None,
                noise_ceiling=None, hatches=None, axis=None, marker="o", markersize=5, legend_title="Model features modality", dodge=False):
    data_filtered = data[data.metric == metric]

    sns.set_style("ticks", {'axes.grid' : True})
    if kind == "bar":
        g = sns.barplot(data_filtered, x=x_variable, order=order, y="value", hue=hue_variable, hue_order=hue_order, palette=palette, err_kws={'linewidth': 0.5}, width=0.95)
    elif kind == "point":
        g = sns.pointplot(data_filtered, x=x_variable, order=order, y="value", hue=hue_variable, hue_order=hue_order, palette=palette,
                          errorbar=None, marker=marker, markersize=markersize, markeredgewidth=1, linestyle="none", ax=axis, dodge=dodge)

    g.legend().remove()
    bbox_extra_artists = None
    if plot_legend:
        # lgd = g.legend(loc='upper left', title="", bbox_to_anchor=(0.05, 0.95), ncol=9)
        lgd = g.legend(ncol=3, title=legend_title)
        bbox_extra_artists = (lgd,)

    if noise_ceiling is not None:
        g.axhline(y=noise_ceiling)

    if hatches is not None:
        for i, thisbar in enumerate(g.patches[:len(hatches)]):
            thisbar.set_hatch(hatches[i])

    g.set(ylim=ylim, ylabel=metric, xlabel='')

    # print(g.get_xticklabels())
    # g.set_xticklabels([label.get_text().split('_')[0] for label in g.get_xticklabels()], rotation=80)

    plt.tight_layout()
    
    return g, data_filtered


def get_short_label_text(label, cut_labels=True):
    text = label.get_text().split('_')[0]
    if cut_labels and (len(text) > 10):
        text = f"{'-'.join(text.split('-')[:-1])}-\n{text.split('-')[-1]}"
    return text
    
def plot_metric_catplot(data, kind="bar", x_variable="model_feat", order=None, row_variable="subject", row_order=None, col_variable=None, hue_variable="model_feat", hue_order=None, metrics=["pairwise_acc_mean"], ylim=(0.5, 1),
                        plot_legend=True, palette=None, noise_ceilings=None, hatches=None, legend_title="Model features modality", height=4, aspect=4, legend_bbox=(0.05,1), rotation=80, cut_labels=True):
    data_filtered = data[data.metric.isin(metrics)]

    sns.set_style("ticks", {'axes.grid' : True})
    g = sns.catplot(data_filtered, kind=kind, x=x_variable, order=order, y="value", row=row_variable, row_order=row_order, col=col_variable, height=height, aspect=aspect, hue=hue_variable, hue_order=hue_order,
                    palette=palette, err_kws={'linewidth': 0.5, 'alpha': 0.99}, width=0.7)
   
    g._legend.remove()
    bbox_extra_artists = None
    if plot_legend:
        # lgd = g.fig.legend(loc='upper left', title="", bbox_to_anchor=(1, 0.9), ncol=2)
        lgd = g.fig.legend(ncol=2, title=legend_title, loc="upper left", bbox_to_anchor=legend_bbox)#, bbox_to_anchor=(0.02, 0.95), ncol=9)
        bbox_extra_artists = (lgd,)

    for i in range(len(g.axes[-1])):
        last_axis = g.axes[-1][i]
        last_axis.set_xticklabels([get_short_label_text(label, cut_labels) for label in last_axis.get_xticklabels()], rotation=rotation)
        
    g.set(ylim=ylim, ylabel="pairwise_acc_mean", xlabel='')
    
    plt.tight_layout()
    
    return g, data_filtered, lgd


FEAT_ORDER = ["vision", "lang", "vision+lang", "matched"]
FEAT_PALETTE = sns.color_palette('Set2')[:4]


def create_result_graph(data, model_feat_order, metrics=["pairwise_acc_captions", "pairwise_acc_images"], hue_variable="features", hue_order=FEAT_ORDER, ylim=None,
                        legend_title="Legend", palette=FEAT_PALETTE, dodge=False, noise_ceilings=None, plot_modality_specific=True,
                       row_variable="metric", row_order=None, col_variable=None, legend_bbox=(0.06,1), height=4.5, aspect=4):
    data_training_mode_full = data[data.training_mode == "modality-agnostic"]
    data_training_mode_captions = data[data.training_mode == "captions"]
    data_training_mode_images = data[data.training_mode == "images"]

    for m_feat in model_feat_order:
        length = len(data_training_mode_full[(data_training_mode_full.model_feat == m_feat) & (data_training_mode_full.metric == metrics[0])])
        expected_num_datapoints = len(SUBJECTS)
        if hue_variable != "features":
            expected_num_datapoints *= len(data[hue_variable].unique())
        assert length == expected_num_datapoints, f"too long or short: {length} (expected: {expected_num_datapoints}) (model_feat: {m_feat}"

    catplot_g, data_plotted, lgd = plot_metric_catplot(data_training_mode_full, order=model_feat_order, metrics=metrics, x_variable="model_feat", legend_title=legend_title, legend_bbox=legend_bbox, height=height, aspect=aspect,
                                                  hue_variable=hue_variable, row_variable=row_variable, row_order=row_order, col_variable=col_variable, hue_order=hue_order, palette=palette, ylim=ylim, noise_ceilings=noise_ceilings)

    if plot_modality_specific:
        _ = plot_metric(data_training_mode_captions, kind="point", order=model_feat_order, metric=metrics[0], x_variable="model_feat", dodge=dodge,
                                      hue_variable=hue_variable, hue_order=hue_order, palette=[(0, 0, 0)], axis=catplot_g.axes[0, 0], marker="o", plot_legend=False, ylim=ylim)
        g, _ = plot_metric(data_training_mode_images, kind="point", order=model_feat_order, metric=metrics[0], x_variable="model_feat", dodge=dodge,
                                      hue_variable=hue_variable, hue_order=hue_order, palette=[(0, 0, 0)], axis=catplot_g.axes[0, 0], marker="x", plot_legend=False, ylim=ylim)
        
        _ = plot_metric(data_training_mode_captions, kind="point", order=model_feat_order, metric=metrics[1], x_variable="model_feat", dodge=dodge,
                                      hue_variable=hue_variable, hue_order=hue_order, palette=[(0, 0, 0)], axis=catplot_g.axes[1, 0], marker="o", plot_legend=False, ylim=ylim)
        _ = plot_metric(data_training_mode_images, kind="point", order=model_feat_order, metric=metrics[1], x_variable="model_feat", dodge=dodge,
                                      hue_variable=hue_variable, hue_order=hue_order, palette=[(0, 0, 0)], axis=catplot_g.axes[1, 0], marker="x", plot_legend=False, ylim=ylim)

        if len(metrics) == 3:
            _ = plot_metric(data_training_mode_captions, kind="point", order=model_feat_order, metric=metrics[2], x_variable="model_feat", dodge=dodge,
                                          hue_variable=hue_variable, hue_order=hue_order, palette=[(0, 0, 0)], axis=catplot_g.axes[2, 0], marker="o", plot_legend=False, ylim=ylim)
            _ = plot_metric(data_training_mode_images, kind="point", order=model_feat_order, metric=metrics[2], x_variable="model_feat", dodge=dodge,
                                          hue_variable=hue_variable, hue_order=hue_order, palette=[(0, 0, 0)], axis=catplot_g.axes[2, 0], marker="x", plot_legend=False, ylim=ylim)
            
        handles, labels = g.get_legend_handles_labels()
        new_labels = ["captions", "images"]
        new_handles = [handles[0], handles[-1]]
        catplot_g.fig.legend(handles=new_handles, labels=new_labels, ncol=2, title="Modality-specific decoders trained on", loc='upper right') #, bbox_to_anchor=(0.05,1)
        
        catplot_g.axes[0, 0].set_title(catplot_g.axes[0,0].title.get_text().split("_")[-1], fontsize=30, y=0.97)
        catplot_g.axes[1, 0].set_title(catplot_g.axes[1,0].title.get_text().split("_")[-1], fontsize=30, y=0.97)
     
        catplot_g.axes[0, 0].set_ylabel('pairwise accuracy')
        catplot_g.axes[1, 0].set_ylabel('pairwise accuracy')

        if len(metrics) == 3:
            catplot_g.axes[2, 0].set_title("overall", fontsize=30, y=0.97)
            catplot_g.axes[2, 0].set_ylabel('pairwise accuracy')

    plt.subplots_adjust(hspace=0.15)
    return catplot_g, lgd


def add_avg_subject(df):
    df_mean = df.copy()
    df_mean["subject"] = "average"
    return pd.concat((df.copy(), df_mean))


def load_results_data(distance_metrics = ["cosine"]):
    results_root_dir = os.path.expanduser(f'~/data/multimodal_decoding/glm/')
    
    metrics = ['rsa', 'rsa_captions', 'rsa_images'] #, 'predictions', 'latents', 'stimulus_ids'
    for dist_metric in distance_metrics:
        metrics.extend([f'acc_{dist_metric}_captions', f'acc_{dist_metric}_images'])# f'acc_{dist_metric}'
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
        data_item = {k: value for k, value in results.items() if k in HP_KEYS}
        data_item["metric"] = "pairwise_acc_mean"
        data_item["value"] = (results["acc_cosine_captions"] + results["acc_cosine_images"]) / 2
        data.append(data_item)


    df = pd.DataFrame.from_records(data)

    df["features"] = df.features.replace({"concat": "vision+lang"})

    df["training_mode"] = df.training_mode.replace({"train": "modality-agnostic", "train_captions": "captions", "train_images": "images"})
    df["mask"] = df["mask"].fillna("whole_brain")

    # create modality-specific decoders with 'matched' features from existing results
    multimodal_models = df[df.features == "vision+lang"].model.unique().tolist()

    data_feat_concat_mod_specific_vision = df[(df.training_mode == "images") & (df.features == "vision") & (df.model.isin(multimodal_models))]
    data_feat_matched_mod_specific_vision = data_feat_concat_mod_specific_vision.copy()
    data_feat_matched_mod_specific_vision["features"] = "matched"
    df = pd.concat((df, data_feat_matched_mod_specific_vision), ignore_index=True)
    
    data_feat_concat_mod_specific_lang = df[(df.training_mode == "captions") & (df.features == "lang") & (df.model.isin(multimodal_models))]
    data_feat_matched_mod_specific_lang = data_feat_concat_mod_specific_lang.copy()
    data_feat_matched_mod_specific_lang["features"] = "matched"
    df = pd.concat((df, data_feat_matched_mod_specific_lang), ignore_index=True)

    df["model_feat"] = df.model + "_" + df.features

    return df
