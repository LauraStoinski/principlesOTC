# The "Fig7_1_run_models.py" already gives me cross validated r2-maps of each property
# But I compute them here, as I accidentally deleted the raw model results on nerz
# I need them for Fig6B

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
from os.path import join as pjoin
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from utils import random_splits, cross_vali_regression_basic, noise_normalize
from nilearn import masking 
import nibabel as nib
import pickle 

# ---------------------------------------------
train = 0.8
n_iter = 10


rootdir =  "/YOUR_PATH/principlesOTC"


betadir =  pjoin(rootdir, "data/whole_brain_FWHM4")
eventdir = pjoin(rootdir, "data")
maskdir = pjoin(rootdir, "data/masks")
resdir = pjoin(rootdir, "results/Fig7b_univariate_r2")
    
regressors = ['animacy', 'curvature', 'size',
       'graspability', 'holdability', 'natural', 'moves', 'precious',
       'be_moved', 'heavy', 'manmade', 'pleasant', 'arousal',
       'agency', 'humanness', 'cuteness', 'hardness',
       'manipulability']


events = pd.read_csv(pjoin(eventdir, "720_events.csv")) 
train_indices, test_indices = random_splits(events, train_size = train, n_iter=n_iter)

for sub in ['01', '02', '03']: 
    mask_brain =  pjoin(maskdir, f"sub-{sub}_space-T1w_brainmask.nii.gz")
    betas = np.load(pjoin(betadir, f"sub{sub}_beta_matrix_whole_brain_means.pkl"), mmap_mode=None, allow_pickle=True) 

    for reg in regressors:
        r2_cross_vali, residuals_cross_vali, residual_counts = cross_vali_regression_basic(
                                                             betas, events, reg, train_indices, test_indices )

  
        with open(pjoin(rootdir, f'data/noise_ceilings/sub-{sub}_nc_n-12_flat.pkl'), "rb") as file:
            noise_ceiling = np.array(pickle.load(file)) 
            
        r2_norm = noise_normalize(r2_cross_vali, noise_ceiling)
        vol_r2 = masking.unmask(r2_norm, mask_brain, order='F')
        nib.save(vol_r2, pjoin(resdir, f"sub{sub}_rsquared_whole_brain_{reg}_noiseNormed_crossvali.nii.gz"))
        
        vol_r2 = masking.unmask(r2_cross_vali, mask_brain, order='F')
        nib.save(vol_r2, pjoin(resdir, f"sub{sub}_rsquared_whole_brain_{reg}_crossvali.nii.gz"))
        
        
        # save residuals for Suppl. Table 1 to 7
        with open(pjoin(resdir, f"sub{sub}_residuals_whole_brain_{reg}_crossvali.pkl"), "wb") as file:
            pickle.dump(residuals_cross_vali, file)

        with open(pjoin(resdir, f"sub{sub}_residual_counts_whole_brain_{reg}_crossvali.pkl"), "wb") as file:
            pickle.dump(residual_counts, file)