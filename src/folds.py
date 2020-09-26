#!/usr/bin/env python

import pandas as pd
from sklearn import model_selection
import config

def create_folds(n_splits):

    df = pd.read_csv(config.INPUT_DATA)
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.KFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    df.to_csv(config.NEW_DATA, index=False)
