# -----------------------------------------------------------------------------
# Fig 7: Step 1:  compute expalined variances for all possible combinations of regresison models
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import os
from os.path import join as pjoin
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from utils_fixed import random_splits, get_intercept, cross_vali_regression


# ---------------------------------------------
train = 0.8
n_iter = 10 

rootdir =  "/YOUR_PATH/principlesOTC"
betadir = pjoin(rootdir, "data/whole_brain_FWHM4")
resdir =  pjoin(rootdir, f"results/Fig7c_hierarchical_partition/train{int(train*100)}")
os.makedirs(resdir, exist_ok=True)


events = pd.read_csv(pjoin(rootdir, "data/720_events.csv"))
properties = ['animacy', 'curvature', 'size', 'natural', 'moves', 'precious', 
              'be_moved', 'heavy', 'pleasant', 'arousal', 'agency', 'humanness', 'cuteness',
              'hardness', 'manipulability']
X = events[properties]
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X, columns=properties)

# Generate all possible model combinations of properties 
all_combinations = [list(comb) for r in range(1, len(properties) + 1) for comb in combinations(properties, r)]
combi_idx = list(range(len(all_combinations)))
labs = pd.DataFrame(zip(combi_idx, all_combinations), columns=['index', 'model'])
labs.to_csv(pjoin(resdir, "hierarchical_partition_models.csv"))

train_indices, test_indices = random_splits(events, train_size = train, n_iter=n_iter)
for sub in ['01', '02', '03']: 
    y = np.load(pjoin(betadir, f"sub{sub}_beta_matrix_whole_brain_means.pkl"), mmap_mode=None, allow_pickle=True)
    y = StandardScaler().fit_transform(y)
    get_intercept(y, sub, resdir, train_indices, test_indices) 

    Parallel(n_jobs=-1)(delayed(cross_vali_regression)(idx, comb, X, y, train_indices, test_indices, resdir, sub) for idx, comb in enumerate(all_combinations))
