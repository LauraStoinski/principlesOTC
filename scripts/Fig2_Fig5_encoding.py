# Object-wise OLS regression of animacy and size, as well as the 13 other object properties
import os
import pickle
from os.path import join as pjoin

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import masking

from utils import linear_regression, noise_normalize


rootdir =  "/YOUR_PATH/principlesOTC"


betadir =  pjoin(rootdir, "data/whole_brain_FWHM4")
eventdir = pjoin(rootdir, "data")
maskdir = pjoin(rootdir, "data/masks")
resdir = pjoin(rootdir, "results/Fig2_Fig5_property_encoding")
os.makedirs(resdir, exist_ok=True)
    
subjects = ['01', '02', '03']
regressors = ['animacy',  'size', 'curvature', 'natural', 'moves', 'precious',
       'be_moved', 'heavy',  'pleasant', 'arousal', 'agency', 
       'humanness', 'cuteness', 'hardness', 'manipulability']

events = pd.read_csv(pjoin(eventdir, "720_events.csv")) 

# compute encoding model
for sub in subjects:
    mask_brain =  pjoin(maskdir, f"sub-{sub}_space-T1w_brainmask.nii.gz")
    betas = np.load(pjoin(betadir, f"sub{sub}_beta_matrix_whole_brain_means.pkl"), mmap_mode=None, allow_pickle=True) 

    for reg in regressors: 
        coefs, r_squared, residuals, y_pred = linear_regression(betas, events, [reg], zscore_y = False, interaction = False)
        
        res = coefs[1] # i=0 is intercept
        vol = masking.unmask(res, mask_brain, order='F')
        nib.save(vol, pjoin(resdir, f"sub{sub}_regression_coefficients_whole_brain_{reg}.nii.gz"))
        
        
        
        with open(pjoin(rootdir, f'data/noise_ceilings/sub-{sub}_nc_n-12_flat.pkl'), "rb") as file:
            noise_ceiling = np.array(pickle.load(file)) 
            
        r2_norm = noise_normalize(r_squared, noise_ceiling)
        vol_r2 = masking.unmask(r2_norm, mask_brain, order='F')
        nib.save(vol_r2, pjoin(resdir, f"sub{sub}_rsquared_whole_brain_{reg}_noiseNormed.nii.gz"))
        
        vol_r2 = masking.unmask(r_squared, mask_brain, order='F')
        nib.save(vol_r2, pjoin(resdir, f"sub{sub}_rsquared_whole_brain_{reg}.nii.gz"))
        