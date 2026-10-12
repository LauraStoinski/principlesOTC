import os
from os.path import join as pjoin
import numpy as np
import pandas as pd
from nilearn import masking
import nibabel as nib
from utils import get_group_means, activity_per_roi


rootdir = "/YOUR_PATH/principlesOTC"

subjects = ["01", "02", "03"]
betadir = pjoin(rootdir, "data/whole_brain_FWHM4")
eventdir = pjoin(rootdir, "data")
maskdir = pjoin(rootdir, "data/masks")
resdir = pjoin(rootdir, "results/Fig3_tripartite")
os.makedirs(resdir, exist_ok=True)

events = pd.read_csv(pjoin(eventdir, "720_events.csv"))

# Compute average beta estimates for (large, small) x (animate, inanimate) across whole brain
ianimals = events.index[(events["animacy_bin"] == "a") & (events["human"] != 1)].tolist() # only non-human animals
iobjects = events.index[(events["animacy_bin"] == "i") & (events["animacy"] <= 2)].tolist() # only very non-animate objects (--> exlcuding moderately animate objects such as plants) 

objects = events.loc[iobjects]
animals = events.loc[ianimals]

for sub in subjects:
    mask_brain = pjoin(maskdir, f"sub-{sub}_space-T1w_brainmask.nii.gz")
    betas = np.load(pjoin(betadir, f"sub{sub}_beta_matrix_whole_brain_means.pkl"), mmap_mode=None, allow_pickle=True)

    assert betas.shape[0] == events.shape[0]

    for ev, group in zip([animals, objects], ["animals", "objects"]):

        means1, means2, sem1, sem2 = get_group_means(betas, ev) # get average response across each of th four groups

        # map form array-space back into brain volume
        vol_means1 = masking.unmask(means1, mask_brain, order="F")
        nib.save(vol_means1, pjoin(resdir, f"sub-{sub}_mean_subset-larger-{group}.nii.gz"))

        vol_means2 = masking.unmask(means2, mask_brain, order="F")
        nib.save(vol_means2, pjoin(resdir, f"sub-{sub}_mean_subset-smaller-{group}.nii.gz"))

        vol_sem1 = masking.unmask(sem1, mask_brain, order="F")
        nib.save(vol_sem1, pjoin(resdir, f"sub-{sub}_SEM_subset-larger-{group}.nii.gz"))

        vol_sem2 = masking.unmask(sem2, mask_brain, order="F")
        nib.save(vol_sem2, pjoin(resdir, f"sub-{sub}_SEM_subset-smaller-{group}.nii.gz"))


# AVERAGE ACTIVITY PER VECTOR-OF-ROI SPHERE
all_df = []
for sub in subjects:
    sani = pjoin(resdir, f"sub-{sub}_mean_subset-smaller-animals.nii.gz")
    sobj = pjoin(resdir, f"sub-{sub}_mean_subset-smaller-objects.nii.gz")
    lani = pjoin(resdir, f"sub-{sub}_mean_subset-larger-animals.nii.gz")
    lobj = pjoin(resdir, f"sub-{sub}_mean_subset-larger-objects.nii.gz")

    sem_sani = pjoin(resdir, f"sub-{sub}_SEM_subset-smaller-animals.nii.gz")
    sem_sobj = pjoin(resdir, f"sub-{sub}_SEM_subset-smaller-objects.nii.gz")
    sem_lani = pjoin(resdir, f"sub-{sub}_SEM_subset-larger-animals.nii.gz")
    sem_lobj = pjoin(resdir, f"sub-{sub}_SEM_subset-larger-objects.nii.gz")

    paths = [lani, sani, lobj, sobj, sem_lani, sem_sani, sem_lobj, sem_sobj]

    condnames1 = ["large animals", "small animals", "large objects", "small objects"]
    condnames = condnames1 + ["sem " + i for i in condnames1]

    df = activity_per_roi(sub, paths, condnames, rootdir) # extract the average per spherical ROI
    all_df.append(df)

out = pd.concat(all_df)

grandMean = out.groupby(["sphere"], as_index=False).mean(numeric_only=True) # compute mean across subjects

# Compute SEM across subjects (as in Konkle & Caramazza)
grandVar = out.groupby(["sphere"], as_index=False).var(numeric_only=True)

for g in condnames1:
    grandMean[f"sem_subj_{g}"] = np.sqrt(grandVar[g] / len(subjects))

out.to_csv(pjoin(resdir, "means_surf_sphere_tripartite.csv"), index=False)
grandMean.to_csv(pjoin(resdir, "Avg_means_surf_sphere_tripartite.csv"), index=False)