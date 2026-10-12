import pickle
import nibabel as nib
from os.path import join as pjoin
from nilearn import image
from nilearn.masking import apply_mask


rootdir =  "/YOUR_PATH/principlesOTC"

ncdir = pjoin(rootdir, "/data/noise_ceilings")
for sub in ['01', '02', '03']: 
    mask =  pjoin(rootdir, f"data/masks/sub-{sub}_space-T1w_brainmask.nii.gz")

    noise_ceiling =  pjoin(ncdir, f"sub-{sub}_nc_n-12.nii.gz")
    
    resamp_mask = nilearn.image.resample_to_img(mask, noise_ceiling, interpolation='nearest')
    
    img = nib.load(noise_ceiling)
    
    img_smoothed =  nilearn.image.smooth_img(img, fwhm = 3) 
    
    masked_data = apply_mask(img, resamp_mask)
    masked_data_smoothed = apply_mask(img_smoothed, resamp_mask)
    
    
    filename = pjoin(ncdir,f"sub-{sub}_nc_n-12_flat.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(masked_data, f)
    
    # also save smoothed version
    filename = pjoin(ncdir,f"sub-{sub}_nc_n-12_flat_smoothed.pkl")
    with open(filename, 'wb') as f:
        pickle.dump(masked_data_smoothed, f)
    
