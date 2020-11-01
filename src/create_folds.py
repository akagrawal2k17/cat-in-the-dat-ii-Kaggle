# create_folds.py
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 12:54:27 2020

@author: aagrawal
"""

# import config for configuration details, pandas and model_delection of scikit-learn
import config

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    
    # Read training data
    df = pd.read_csv(config.TRAINING_FILE)
    
    # We create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    #fetch labels
    y = df.target.values
    
    # inititate the kfold class from model_selection
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    # fill the new kfold column
    for f,(train_idx_, test_idx_) in enumerate(kf.split(X=df,y=y)):
        df.loc[test_idx_,"kfold"] = f
        
    # save the new csv with kfold column
    df.to_csv(config.TRAIN_FOLDS_FILE, index=False)



