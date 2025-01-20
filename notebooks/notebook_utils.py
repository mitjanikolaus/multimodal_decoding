import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data import DEFAULT_FEATURES, DEFAULT_VISION_FEATURES, DEFAULT_LANG_FEATURES
from eval import ACC_MODALITY_AGNOSTIC, ACC_CROSS_IMAGES_TO_CAPTIONS, ACC_CROSS_CAPTIONS_TO_IMAGES, \
    calc_all_pairwise_accuracy_scores
from utils import SUBJECTS, RIDGE_DECODER_OUT_DIR, MODE_AGNOSTIC, MOD_SPECIFIC_CAPTIONS, MOD_SPECIFIC_IMAGES
from analyses.decoding.ridge_regression_decoding import ACC_CAPTIONS, ACC_IMAGES, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST

from tqdm import tqdm
from glob import glob
import pickle

HP_KEYS = ["alpha", "model", "subject", "features", "test_features", "vision_features", "lang_features",
           "training_mode", "mask",
           "num_voxels", "surface", "resolution"]
METRIC_NAMES = {"acc_cosine": "pairwise_acc", "acc_cosine_captions": "pairwise_acc_captions",
                "acc_cosine_images": "pairwise_acc_images"}

ACC_MEAN = "pairwise_acc_mean"
ACC_CROSS_MEAN = "pairwise_acc_cross_mean"


def plot_metric(data, kind="bar", x_variable="model_feat", order=None, hue_variable="model_feat", hue_order=None,
                metric="pairwise_acc_mean", ylim=(0.5, 1), plot_legend=True, palette=None,
                noise_ceiling=None, hatches=None, axis=None, marker="o", markersize=5,
                legend_title="Model features modality", dodge=False):
    data_filtered = data[data.metric == metric]

    sns.set_style("ticks", {'axes.grid': True})
    palette = palette[:len(hue_order)]
    if kind == "bar":
        g = sns.barplot(data_filtered, x=x_variable, order=order, y="value", hue=hue_variable, hue_order=hue_order,
                        palette=palette, err_kws={'linewidth': 0.5}, width=0.95)
    elif kind == "point":
        g = sns.pointplot(data_filtered, x=x_variable, order=order, y="value", hue=hue_variable, hue_order=hue_order,
                          palette=palette,
                          errorbar=None, marker=marker, markersize=markersize, markeredgewidth=1, linestyle="none",
                          ax=axis, dodge=dodge)

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
    if cut_labels and (len(text) > 10) and ('-' in text):
        text = f"{'-'.join(text.split('-')[:-1])}-\n{text.split('-')[-1]}"
    return text


def plot_metric_catplot(data, kind="bar", x_variable="model_feat", order=None, row_variable="subject", row_order=None,
                        col_variable=None, hue_variable="model_feat", hue_order=None, metrics=["pairwise_acc_mean"],
                        ylim=(0.5, 1),
                        plot_legend=True, palette=None, noise_ceilings=None, hatches=None,
                        legend_title="Model features modality", height=4, aspect=4, legend_bbox=(0.05, 1), rotation=80,
                        cut_labels=True, shorten_label_texts=True):
    data_filtered = data[data.metric.isin(metrics)]

    sns.set_style("ticks", {'axes.grid': True})
    palette = palette[:len(hue_order)]
    g = sns.catplot(data_filtered, kind=kind, x=x_variable, order=order, y="value", row=row_variable,
                    row_order=row_order, col=col_variable, height=height, aspect=aspect, hue=hue_variable,
                    hue_order=hue_order,
                    palette=palette, err_kws={'linewidth': 0.5, 'alpha': 0.99}, width=0.7)

    if g._legend is not None:
        g._legend.remove()
    bbox_extra_artists = None
    lgd = None
    if plot_legend:
        # lgd = g.fig.legend(loc='upper left', title="", bbox_to_anchor=(1, 0.9), ncol=2)
        lgd = g.fig.legend(ncol=2, title=legend_title, loc="upper left",
                           bbox_to_anchor=legend_bbox)  # , bbox_to_anchor=(0.02, 0.95), ncol=9)
        bbox_extra_artists = (lgd,)

    for i in range(len(g.axes[-1])):
        last_axis = g.axes[-1][i]
        if shorten_label_texts:
            last_axis.set_xticklabels(
                [get_short_label_text(label, cut_labels) for label in last_axis.get_xticklabels()])
        last_axis.tick_params(axis='x', rotation=rotation)

    g.set(ylim=ylim, ylabel="pairwise_acc_mean", xlabel='')

    plt.tight_layout()

    return g, data_filtered, lgd


FEAT_ORDER = ["vision", "lang", "vision+lang", "matched"]
FEAT_PALETTE = sns.color_palette('Set2')
PALETTE_BLACK_ONLY = [(0, 0, 0)] * 10


def create_result_graph(data, x_variable="model_feat", order=None,
                        metrics=["pairwise_acc_captions", "pairwise_acc_images"],
                        hue_variable="features", hue_order=FEAT_ORDER, ylim=None,
                        legend_title="Legend", palette=FEAT_PALETTE, dodge=False, noise_ceilings=None,
                        plot_modality_specific=True,
                        row_variable="metric", row_order=None, col_variable=None, legend_bbox=(0.06, 0.97),
                        legend_2_bbox=(0.99, 0.97), height=4.5, row_title_height=0.85, aspect=4,
                        verify_num_datapoints=True, plot_legend=True, shorten_label_texts=True):
    data_training_mode_full = data[data.training_mode == "modality-agnostic"]
    data_training_mode_captions = data[data.training_mode == "captions"]
    data_training_mode_images = data[data.training_mode == "images"]

    for mode in ["modality-agnostic", "captions", "images"]:
        data_mode = data[data.training_mode == mode]
        for x_variable_value in order:
            length = len(data_mode[(data_mode[x_variable] == x_variable_value) & (data_mode.metric == metrics[0])])
            expected_num_datapoints = len(SUBJECTS)
            if hue_variable != "features":
                expected_num_datapoints *= len(data[hue_variable].unique())
            if (length > 0) and (length != expected_num_datapoints):
                message = f"unexpected number of datapoints: {length} (expected: {expected_num_datapoints}) ({x_variable}: {x_variable_value} {mode}"
                if verify_num_datapoints:
                    raise RuntimeError(message)
                else:
                    print(f"Warning: {message}")

    catplot_g, data_plotted, lgd = plot_metric_catplot(data_training_mode_full, order=order, metrics=metrics,
                                                       x_variable=x_variable, legend_title=legend_title,
                                                       legend_bbox=legend_bbox, height=height, aspect=aspect,
                                                       hue_variable=hue_variable, row_variable=row_variable,
                                                       row_order=row_order, col_variable=col_variable,
                                                       hue_order=hue_order, palette=palette, ylim=ylim,
                                                       noise_ceilings=noise_ceilings, plot_legend=plot_legend,
                                                       shorten_label_texts=shorten_label_texts)

    if plot_modality_specific:
        first_metric_graph_mod_specific_1 = None

        for m, metric in enumerate(metrics):
            g1, _ = plot_metric(data_training_mode_captions, kind="point", order=order, metric=metrics[m],
                                x_variable="model_feat", dodge=dodge,
                                hue_variable=hue_variable, hue_order=hue_order, palette=PALETTE_BLACK_ONLY,
                                axis=catplot_g.axes[m, 0], marker="o", plot_legend=False, ylim=ylim)
            g2, _ = plot_metric(data_training_mode_images, kind="point", order=order, metric=metrics[m],
                                x_variable="model_feat", dodge=dodge,
                                hue_variable=hue_variable, hue_order=hue_order, palette=PALETTE_BLACK_ONLY,
                                axis=catplot_g.axes[m, 0], marker="x", plot_legend=False, ylim=ylim)
            if m == 0:
                first_metric_graph_mod_specific_1 = g1

        handles, labels = first_metric_graph_mod_specific_1.get_legend_handles_labels()

        if len(handles) > 0:
            new_labels = ["captions", "images"]
            new_handles = [handles[-4], handles[-1]]
            catplot_g.fig.legend(handles=new_handles, labels=new_labels, ncol=2,
                                 title="Modality-specific decoders trained on", loc='upper right',
                                 bbox_to_anchor=legend_2_bbox)

    for m, metric in enumerate(metrics):
        catplot_g.axes[m, 0].set_title(
            metrics[m].replace("pairwise_acc_mean", "").replace("pairwise_acc_", "").replace("_", "-"), fontsize=35,
            y=row_title_height)
        catplot_g.axes[m, 0].set_ylabel('pairwise accuracy')

    plt.subplots_adjust(hspace=0.15)
    return catplot_g, lgd


def add_avg_subject(df):
    df_mean = df.copy()
    df_mean["subject"] = "average"
    return pd.concat((df.copy(), df_mean))


METRICS_BASE = [ACC_MODALITY_AGNOSTIC, ACC_CAPTIONS, ACC_IMAGES, ACC_CROSS_IMAGES_TO_CAPTIONS,
                ACC_CROSS_CAPTIONS_TO_IMAGES, ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST]
METRICS_ERROR_ANALYSIS = ['predictions', 'latents', 'stimulus_ids', 'stimulus_types']
METRICS_IMAGERY = METRICS_ERROR_ANALYSIS + [ACC_IMAGERY, ACC_IMAGERY_WHOLE_TEST, 'imagery_predictions',
                                            'imagery_latents']


def update_acc_scores(results, metric="cosine", standardize_predictions=True, standardize_targets=False,
                      norm_imagery_preds_with_test_preds=False):
    results.update(
        calc_all_pairwise_accuracy_scores(
            results["latents"], results["predictions"], results["stimulus_types"],
            imagery_latents=results["imagery_latents"] if "imagery_latents" in results else None,
            imagery_predictions=results["imagery_predictions"] if "imagery_predictions" in results else None,
            metric=metric, standardize_predictions=standardize_predictions, standardize_latents=standardize_targets,
            norm_imagery_preds_with_test_preds=True
        )
    )
    return results


def load_results_data(models, metrics=METRICS_BASE, recompute_acc_scores=False, pairwise_acc_metric="cosine",
                      standardize_predictions=True, standardize_targets=False,
                      norm_imagery_preds_with_test_preds=False):
    data = []

    result_files = sorted(glob(f"{RIDGE_DECODER_OUT_DIR}/*/*/*/results.p"))
    for result_file_path in tqdm(result_files):
        results = pickle.load(open(result_file_path, 'rb'))
        if results['model'] in models:
            if recompute_acc_scores:
                results = update_acc_scores(results, metric=pairwise_acc_metric,
                                            standardize_predictions=standardize_predictions,
                                            standardize_targets=standardize_targets,
                                            norm_imagery_preds_with_test_preds=norm_imagery_preds_with_test_preds)

            for metric in metrics:
                if metric in results.keys():
                    data_item = {k: value for k, value in results.items() if k in HP_KEYS}
                    data_item["metric"] = metric
                    data_item["value"] = results[metric]
                    data.append(data_item)
            data_item_acc_mean = {k: value for k, value in results.items() if k in HP_KEYS}
            data_item_acc_mean["metric"] = ACC_MEAN
            data_item_acc_mean["value"] = (results[ACC_CAPTIONS] + results[ACC_IMAGES]) / 2
            data.append(data_item_acc_mean)

            # data_item_acc_cross_mean = {k: value for k, value in results.items() if k in HP_KEYS}
            # data_item_acc_cross_mean["metric"] = ACC_CROSS_MEAN
            # data_item_acc_cross_mean["value"] = (results[ACC_CROSS_IMAGES_TO_CAPTIONS] + results[ACC_CROSS_CAPTIONS_TO_IMAGES]) / 2
            # data.append(data_item_acc_cross_mean)

    df = pd.DataFrame.from_records(data)

    if "test_features" in df.columns:
        df = df[(df.test_features == df.features) | df.test_features.isna()].copy()

    df["training_mode"] = df.training_mode.replace(
        {MODE_AGNOSTIC: "modality-agnostic", MOD_SPECIFIC_CAPTIONS: "captions", MOD_SPECIFIC_IMAGES: "images"})

    if "surface" in df.columns:
        df["surface"] = df.surface.fillna(False)
    else:
        df["surface"] = False

    df["vision_features"] = df.vision_features.replace(
        {"visual_feature_mean": "vision_features_mean", "visual_feature_cls": "vision_features_cls"})

    # imagebind only supports extraction of cls features
    df.loc[df.model == "imagebind", "lang_features"] = "lang_features_cls"

    # we currently always compute mean features from language models 
    df.loc[df.model.isin(
        ["bert-base-uncased", "bert-large-uncased", "llama2-7b", "llama2-13b", "mistral-7b", "mixtral-8x7b",
         "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]), "lang_features"] = "lang_features_mean"

    # update unimodal feat values for fused feats of multimodal models:
    df.loc[df.features.isin(["fused_mean", "fused_cls"]), "lang_features"] = "n_a"
    df.loc[df.features.isin(["fused_mean", "fused_cls"]), "vision_features"] = "n_a"

    # default values for unimodal models:
    df.loc[df.model.isin(
        ["bert-base-uncased", "bert-large-uncased", "llama2-7b", "llama2-13b", "mistral-7b", "mixtral-8x7b",
         "gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"]), "vision_features"] = "n_a"
    df.loc[df.model.isin(["vit-b-16", "vit-l-16", "resnet-18", "resnet-50", "resnet-152", "dino-base", "dino-large",
                          "dino-giant"]), "lang_features"] = "n_a"

    # df["lang_features"] = df["lang_features"].fillna("unk")

    df["mask"] = df["mask"].fillna("whole_brain")
    df["mask"] = df["mask"].apply(lambda x: x.replace("p_values_", "").replace(".p", ""))

    df["model_feat"] = df.model + "_" + df.features

    return df


def get_data_default_feats(data):
    data_default_feats = data.copy()
    for model in data.model.unique():
        default_feats = DEFAULT_FEATURES[model]
        default_vision_feats = DEFAULT_VISION_FEATURES[model]
        default_lang_feats = DEFAULT_LANG_FEATURES[model]
        data_default_feats = data_default_feats[
            ((data_default_feats.model == model) & (data_default_feats.features == default_feats) & (
                        data_default_feats.vision_features == default_vision_feats) & (
                         data_default_feats.lang_features == default_lang_feats)) | (data_default_feats.model != model)
            ]

    return data_default_feats
