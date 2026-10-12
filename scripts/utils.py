import pickle
import numpy as np
import pandas as pd
import nibabel as nib

from os.path import join as pjoin
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

############################################################################################################################

# Fig1 and Fig5: Encoding models to create regression slope topographies

############################################################################################################################

def columnwise_correlations(array1, array2):
    mean1 = np.mean(array1, axis=0)
    mean2 = np.mean(array2, axis=0)
    
    
    array1_centered = array1 - mean1
    array2_centered = array2 - mean2
    
    numerator = np.sum(array1_centered * array2_centered, axis=0)
    
    norm1 = np.sqrt(np.sum(array1_centered**2, axis=0))
    norm2 = np.sqrt(np.sum(array2_centered**2, axis=0))
    denominator = norm1 * norm2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        correlations = numerator / denominator
    
    # Handle divisions by zero or invalid results
    correlations = np.where(np.isfinite(correlations), correlations, 0)
    
    return correlations

def linear_regression(y, events, regressors, zscore_y=False, interaction=False, fit_intercept=True):
    """
    OLS linear regression
    """
    
    
    # check if predictors are passed as dataframe or array
    if isinstance(events, pd.DataFrame):
        x = events[regressors].to_numpy(copy=True)
    else:
        x = np.asarray(events)

    if x.ndim == 1: # if only one predictor, reshape
        x = x[:, None]

    # optional: compute interaction between predictors
    if interaction:
        poly = PolynomialFeatures(interaction_only=True, include_bias=False)
        x = poly.fit_transform(x)

    x = StandardScaler().fit_transform(x) # standardize predictors 

    # default: fit intercept
    if fit_intercept:
        x = np.column_stack([np.ones(x.shape[0]), x])

    # Optional: standardize y
    if zscore_y:
        y = StandardScaler().fit_transform(y)

    coefs = np.linalg.lstsq(x, y, rcond=None)[0]
    y_pred = x @ coefs
    residuals = y - y_pred

    sst = np.sum((y - np.mean(y, axis=0))**2, axis=0)
    rss = np.sum(residuals**2, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = 1 - rss / sst


    return coefs, r_squared, residuals, y_pred



############################################################################################################################

# Fig 3 (Tripartite): Response profiles of large and small animals and objects

############################################################################################################################

def get_quantile_idx(df, variable = 'size', lower = 0.25, upper = 0.75):
    # get cut offs of the 25% largest and smallest animals or objects
    qlower = np.quantile(df[variable], lower)
    qupper = np.quantile(df[variable], upper)   

    i1 = []
    i2 = []

    for i, val in enumerate(df[variable]):
        row_idx = df.index[i]
        if val >= qupper:  
            i1.append(row_idx)
        elif val <= qlower:
            i2.append(row_idx)
        
    # ensure they both have the same lenght:
    min_len = min(len(i1), len(i2))
    i1 = i1[:min_len]
    i2 = i2[:min_len]
    
    return i1, i2

def get_group_means(betas, events):
    
    # i1 is larger, i2 is smaller
    i1, i2 =  get_quantile_idx(events, 'size', lower = 0.25, upper = 0.75) 

    print(len(i1), len(i2))
    group1 = betas[i1, :]
    group2 = betas[i2, :] 

    means1 = np.mean(group1, axis = 0)
    means2 = np.mean(group2, axis = 0)

    # compute voxel-wise mean per group 
    assert len(means1) == group1.shape[1], "mean matrix has wrong shape " + str(means1.shape)
    assert means1.shape == means2.shape, "mean matrices have different shape"

    # compute the pooled variance
    var1 = np.var(group1,  axis = 0)
    var2 = np.var(group2,  axis = 0)
    n1 = group1.shape[0]
    n2 = group2.shape[0]

    # compute the SEM 
    sem1 = np.sqrt(var1/n1) 
    sem2 = np.sqrt(var2/n2) 
        
    return  means1, means2, sem1, sem2


############################################################################################################################

# Fig 7: Hierarchical partition analysis

############################################################################################################################

def random_splits(events, train_size, n_iter):
    """Randomly assign train and test splits with specified train size and a seed for reproducibility.
    e.g. train_size = 0.8 for 80% train data
    n_iter = 10 for 10 iterations of validation
    """
    seeds = np.arange(0, n_iter)
    train_indices = []
    test_indices = []
    for seed in seeds: 
        np.random.seed(seed)
        
        # Shuffle the indices
        indices = np.random.permutation(events.index)
        
        # Calculate the number of training samples
        n_train = int(len(indices) * train_size)
        
        # Split the indices into train and test sets
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_indices.append(train_idx)
        test_indices.append(test_idx)
    
    return train_indices, test_indices

def get_intercept(y, sub, resdir, train_indices, test_indices):
    """Save intercept-only test R² averaged over the property models' splits."""
    r2_scores = []
    for train_idx, test_idx in zip(train_indices, test_indices):
        y_train, y_test = y[train_idx], y[test_idx]
        intercept = np.mean(y_train, axis=0)
        residuals_test = y_test - intercept
        y_mean_test = np.mean(y_test, axis=0)
        SST_test = np.sum((y_test - y_mean_test) ** 2, axis=0)
        RSS_test = np.sum(residuals_test ** 2, axis=0)
        test_r2 = 1 - (RSS_test / SST_test)
        r2_scores.append(test_r2)

    mean_r2 = np.mean(r2_scores, axis=0)
    with open(pjoin(resdir, f'sub-{sub}_r_squared_intercept.pkl'), 'wb') as f:
        pickle.dump(mean_r2, f)
     
     
def cross_vali_regression(model_index, combination, X_all, y, train_indices, test_indices, resdir, sub, fit_intercept=True):
    "cross validated linear regression for special case of for hierarchical partition"
    r2_scores = []
    X = X_all[combination].to_numpy()# only subsample required regressors
    for train_idx, test_idx in zip(train_indices, test_indices):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # inlcude intercept
        if fit_intercept:
            X_train = np.hstack([np.ones((X_train_raw.shape[0], 1)), X_train_raw])
            X_test = np.hstack([np.ones((X_test_raw.shape[0], 1)), X_test_raw])
        else:
            X_train = X_train_raw
            X_test = X_test_raw
            
        #coefs, _, _,  _, _,  = linear_regression(y_train, X_train, combination, zscore_x = False,  zscore_y = False) # dont zscore_x and y, as i already did it for train and test_x together 
        coefs = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
        y_pred_test = X_test @ coefs
        
        # compute propert r2
        residuals_test = y_test - y_pred_test
        y_mean_test = np.mean(y_test, axis=0)
        SST_test = np.sum((y_test - y_mean_test) ** 2, axis=0)
        RSS_test = np.sum(residuals_test ** 2, axis=0)
        test_r2 = 1 - (RSS_test / SST_test)
        
        r2_scores.append(test_r2)
        
        # also compute pseudo r2
        #test_r2_pseudo = (columnwise_correlations(y_test, y_pred_test))**2
        

    mean_r2 = np.mean(r2_scores, axis=0)
    sd_r2 = np.std(r2_scores, axis=0)
  
    with open(pjoin(resdir, f'sub-{sub}_cross_vali_r_squared_model{model_index}.pkl'), 'wb') as f:
        pickle.dump(mean_r2, f)
    with open(pjoin(resdir, f'sub-{sub}_cross_vali_r_squared_SD_model{model_index}.pkl'), 'wb') as f:
        pickle.dump(sd_r2, f)
        
        
        
############################################################################################################################

# Other helper functions

############################################################################################################################
def activity_per_roi(subject, paths, condnames, rootdir):
    """
    return mean activity over spheres + the SEM*1.96 of voxel
    """
    assert len(paths) == len(condnames), "conditions and condition names have different length"
    

    spheres = np.load(pjoin(rootdir, "data/vector-of-roi/", f"sub{int(subject):02d}_surf_spheres.npz"))
    n_spheres = len(spheres.files)
    
    subid = [subject]* n_spheres
    sphereid = [i + 1 for i in range(n_spheres)]
    df = pd.DataFrame(list(zip(subid, sphereid)), columns = ["subject", "sphere"])

    for impath, cond in zip(paths, condnames):
        if isinstance(impath, nib.nifti1.Nifti1Image):
            img = impath   
        else:
            img = nib.load(impath) 

        arr = np.array(img.dataobj)

        input = arr
        
        input = np.swapaxes(arr,0,2) 

        res_col = []
        sem_col = []
        for roi in range(n_spheres):
           
            betas = input
            cur_sphere = spheres[list(spheres.keys())[roi]]


            values = betas[cur_sphere > 0]
            values = values.tolist()
            mean = np.sum(values) / len(values)
            var = np.var(values)
            sem = np.sqrt(var/len(values)) * 1.96
            res_col.append(mean)
            sem_col.append(sem)
        
        df[cond] = res_col
        df[f"sem_{cond}_voxel"] = sem_col 
    
    return df

def noise_normalize(r2, noise_ceiling):
    noise_ceiling = noise_ceiling / 100
    noise_ceiling = np.where(noise_ceiling < 0.05, np.nan, noise_ceiling) # set voxel with low noise ceiling to nan
    nan_mask = np.isnan(noise_ceiling)
    array_norm = np.zeros_like(r2)
    r2_normed = np.divide(r2, noise_ceiling, out=array_norm, where=~nan_mask)
    return r2_normed