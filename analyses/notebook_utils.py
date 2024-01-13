import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import FEATURES_DIR

from glob import glob
import pickle
from tqdm import tqdm


def load_results_data(training_mode, regression_models, distance_metrics, subjects):
    results_root_dir = os.path.expanduser(f'~/data/multimodal_decoding/glm/{training_mode}')
    
    metrics = ['val_loss', 'val_rsa', 'rsa', 'predictions', 'latents', 'stimulus_ids']
    for dist_metric in distance_metrics:
        metrics.extend([f'acc_{dist_metric}', f'acc_{dist_metric}_captions', f'acc_{dist_metric}_images', f'val_acc_{dist_metric}'])         
    
    all_dfs = []
    
    for regression_model in regression_models:
        data = []
        for subjectidx, subject in enumerate(subjects):
            results_file_path = os.path.join(results_root_dir, regression_model, subject)
            
            result_files = sorted(glob(f"{results_file_path}/*/*/results.p"))
            for result_file_path in result_files:
                if "full_train" in result_file_path:
                    model_name = result_file_path.split(subject)[1][1:].split('/')[0]
                    hp_str = result_file_path.split(model_name)[1][1:]
                    hp_str = hp_str[:hp_str.index('/')]
                elif "best_hp" in result_file_path:
                    # TODO
                    continue
                else:
                    model_name = result_file_path.split(subject)[1][1:].split('/')[0]
                    hp_str = result_file_path.split(model_name)[1][1:]
                    hp_str = hp_str.split("fold")[0]
                
                results = pickle.load(open(result_file_path, 'rb'))

                for key in results.keys():
                    if key in metrics:
                        data.append({
                            "subject": subject,
                            "hp": hp_str,
                            "model": model_name,
                            "metric": key,
                            "value": results[key],
                            "regression_model": regression_model,
                        })   

        df = pd.DataFrame.from_records(data)
        
        df_mean = df.copy()
        df_mean["subject"] = "average"
        df = pd.concat((df, df_mean))
        all_dfs.append(df)
    
    all_data = pd.concat(all_dfs)

    all_data["hp_str"] = all_data.regression_model + '_' + all_data.hp

    return all_data
    
