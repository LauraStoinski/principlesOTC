import pandas as pd
import pickle
import os
import numpy as np
from os.path import join as pjoin
from utils import noise_normalize  
from tqdm import tqdm

train = 0.8
n_iter = 10

def load_r2(directory, indices, subject):
    """Loads cross-validated r^2 pickles for all given model indices."""
    r_squared = {}
    for model_id in indices:
        filename = f"sub-{subject}_cross_vali_r_squared_model{model_id}.pkl"
        with open(pjoin(directory, filename), "rb") as file:
            r_squared[model_id] = np.array(pickle.load(file))
    return r_squared 

def parse_model(model_str):
    """
    from the "hierarchical_partition_models.csv, trasnform the model design into list
    """
    # Remove outer brackets [ ] if present
    model_str = model_str.strip("[]")
    # Remove any single quotes
    model_str = model_str.replace("'", "")
    # Split on commas, strip extra whitespace
    model_list = [m.strip() for m in model_str.split(",") if m.strip()]
    return model_list

rootdir =  "/YOUR_PATH/principlesOTC"
datadir = pjoin(rootdir, f"results/Fig7c_hierarchical_partition/train{int(train*100)}")

resdir = pjoin(datadir, "incrementals")
os.makedirs(resdir, exist_ok=True)

df = pd.read_csv(pjoin(datadir, "hierarchical_partition_models.csv"))
df['model'] = df['model'].apply(parse_model)

subjects = ['01', '02', '03'] 
properties = [
    'animacy', 'curvature', 'size', 
    'moves', 'precious', 'be_moved', 'pleasant', 'heavy', 'natural', 
    'arousal', 'agency', 'humanness', 'cuteness', 
    'hardness', 'manipulability'
] 
n_models = df['index'].iloc[-1]
assert len(df['model'].iloc[-1]) == len(properties), \
    "The last model should be the full model with all properties."


for sub in tqdm(subjects, desc="Processing Subjects"):
    
    # preload the r2 of all models
    r_squared = load_r2(datadir, df['index'], sub)
    
    # Load intercept model r2 and noise ceiling
    with open(pjoin(datadir, f'sub-{sub}_r_squared_intercept.pkl'), "rb") as file:
        intercept = np.array(pickle.load(file))
    with open(pjoin(rootdir, f'data/noise_ceilings/sub-{sub}_nc_n-12_flat_smoothed.pkl'), "rb") as file:
        noise_ceiling = np.array(pickle.load(file)) 
    
    
    for predictor in tqdm(properties, desc=f"Subject {sub} - Processing Predictors", leave=False):
        incremental_contributions = []
        # Find all model indices that contain this predictor
        pred_models = df[df['model'].apply(lambda x: predictor in x)]['index'].tolist()
        
        if not pred_models:
            print(f"[DEBUG] No models found containing '{predictor}'. Skipping.")
            continue
        
        pred_models_set = set(pred_models)
        
        # First, compare univariate model with intercept model
        single_pred_rows = df[df['model'].apply(lambda x: set(x) == {predictor})]
        assert single_pred_rows.empty == False, "[ERROR] the univariate model is missing"
            
        uni_model_idx = single_pred_rows['index'].iloc[0]
        incremental_contributions.append(r_squared[uni_model_idx] - intercept)
        pred_models_set.remove(uni_model_idx)
    
            
        # For each multi-predictor model containing this predictor, 
            # find the matched model that is the same set minus this predictor.
        for model_with_id in pred_models_set:
            model_with = set(df.loc[df['index'] == model_with_id, 'model'].iloc[0])
            model_without = model_with - {predictor}
            assert len(model_with) - 1 == len(model_without), "Mismatch in model sets!"
            
            rows_no_pred = df[df['model'].apply(lambda x: set(x) == model_without)]
            if rows_no_pred.empty:
                print(f"[DEBUG] Could not find 'model_without' for {model_with}. Skipping.")
                continue
            
            # assure that only one match found
            model_without_id = rows_no_pred['index'].iloc[0]
            assert len(rows_no_pred) == 1, "[ERROR] more than one model-without-predictor matching the current model-with-property "
            
            # Compute the incremental: r2(model with) - r2(model without)
            increm = r_squared[model_with_id] - r_squared[model_without_id]
            incremental_contributions.append(increm)
            
        # compute mean of the incrementals across all models
        stacked = np.vstack(incremental_contributions)
        avg_array = np.mean(stacked, axis=0)
        
        output_path = pjoin(resdir, f"sub{sub}_{predictor}_contributions.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(avg_array, file)
        
        array_norm = noise_normalize(avg_array, noise_ceiling)
        output_path2 = pjoin(resdir, f"sub{sub}_{predictor}_contributions_noiseNormed.pkl")
        with open(output_path2, 'wb') as file:
            pickle.dump(array_norm, file)

    # also noise-normalize the full model
    r2 = np.load(pjoin(datadir, f"sub-{sub}_cross_vali_r_squared_model{n_models}.pkl"), mmap_mode=None, allow_pickle=True) 
    r2 = r2.squeeze()
    full_norm = noise_normalize(r2, noise_ceiling)

    output_path3 = pjoin(datadir, f"sub-{sub}_cross_vali_r_squared_model{n_models}_noiseNormed.pkl")
    with open(output_path3, 'wb') as file:
        pickle.dump(full_norm, file)