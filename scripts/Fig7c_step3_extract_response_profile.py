import nibabel as nib
import nilearn
from nilearn import image, masking
import numpy as np
import pandas as pd

from os.path import join as pjoin
from utils import activity_per_roi

rootdir = "/YOUR_PATH/principlesOTC"
datadir =pjoin(rootdir, f"results/Fig7c_hierarchical_partition/train80")

model_info = pd.read_csv(pjoin(datadir, "hierarchical_partition_models.csv")) 
maxmodel = model_info["index"].max() # full model with all regressors 
properties = [
    "animacy", 
    "curvature", "size", "natural", "moves", "precious", "be_moved",
    "heavy", "pleasant", "arousal", "agency", "humanness",
    "cuteness", "hardness", "manipulability"
]
    
######################
#version = ""
version = "_noiseNormed"
subjects = [1, 2, 3]
#########################

# First, transform pkl to nifti
for sub in subjects:
    ref = nib.load(pjoin(rootdir, f"results/Fig2_Fig5_property_encoding/sub0{sub}_regression_coefficients_whole_brain_animacy.nii.gz"))
    mask = nilearn.image.resample_to_img(pjoin(rootdir, f"data/masks/sub-0{sub}_space-T1w_brainmask.nii.gz"), ref, interpolation='nearest')
 
    for prop in properties: 
        cur_arr = np.load(pjoin(datadir, f"incrementals/sub0{sub}_{prop}_contributions{version}.pkl"), mmap_mode=None, allow_pickle=True) 
        cur_arr = cur_arr.squeeze()

        vol = nilearn.masking.unmask(cur_arr, mask, order='F')
        nib.save(vol, pjoin(datadir, f"incrementals/sub0{sub}_{prop}_contributions{version}.nii.gz"))
 
    # also do this for total variance (last model) and the noise normalize version
    cur_arr = np.load(pjoin(datadir, f"sub-0{sub}_cross_vali_r_squared_model{maxmodel}{version}.pkl"), mmap_mode=None, allow_pickle=True) 
    cur_arr = cur_arr.squeeze()
    vol2 = nilearn.masking.unmask(cur_arr, mask, order='F')
    nib.save(vol2,  pjoin(datadir, f"sub0{sub}_cross_vali_r_squared_model{maxmodel}_totalVar{version}.nii.gz"))
    
# Second, extract Vector of Roi Profiles

all_df = []
labels = ['totalVar'] + properties
for subject in subjects:

    totalVar = pjoin(datadir, f"sub0{subject}_cross_vali_r_squared_model{maxmodel}_totalVar{version}.nii.gz")
    paths = [totalVar]

    for prop in properties:
        cur_path = pjoin(datadir, f"incrementals/sub0{subject}_{prop}_contributions{version}.nii.gz")
        paths.append(cur_path)

        
    df = activity_per_roi(subject, paths, labels, rootdir)
    all_df.append(df)


out = pd.concat(all_df)
out.to_csv(pjoin(datadir, f"weights_per_surf_sphere_hierarchical_partition{version}.csv"), index = False)

grandMean = out.groupby([ 'sphere'], as_index=False).mean() 
grandMean.to_csv(pjoin(datadir, f"Avg_weights_per_surf_sphere_hierarchical_partition{version}.csv"), index = False)