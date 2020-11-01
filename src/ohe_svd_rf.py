# ohe_svd_rf.py
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:28:14 2020

@author: aagrawal
"""

import config

import pandas as pd

from sklearn import metrics, preprocessing, ensemble, decomposition
from scipy import sparse

def run(fold):
    
    # load full training data with folds
    df = pd.read_csv(config.TRAIN_FOLDS_FILE)
    
    # all columns are features except id, target and kfold columns
    features = [x for x in df.columns if x not in ("id", "target", "kfold")]
    
    
    # fill all NaN values to "NONE" and convert the columns values to string.
    # It does not matter since all features are categorical so we can convert them to string    
    for col in features:
        df.loc[:, col] = df.loc[:, col].astype(str).fillna("NONE")  
        
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop= True) 
    
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # initialize one hot encoder
    ohe = preprocessing.OneHotEncoder()
    
    # fit ohe on training + validiation  features
    full_data = pd.concat([df_train[features],df_valid[features]], axis = 0)    
    ohe.fit(full_data[features])    
    # transform training data
    X_train = ohe.transform(df_train[features])    
    # transform validation data
    X_valid = ohe.transform(df_valid[features])
    
    # initilaize truncated svd
    # we are reducing data to 120 components    
    svd = decomposition.TruncatedSVD(n_components=120)
    
    # fit svd on full sparse trainign data
    full_sparse = sparse.vstack((X_train,X_valid))
    svd.fit(full_sparse)
    
    #transform sparse training data
    X_train = svd.transform(X_train)
    
    # transform sparse validation data
    X_valid = svd.transform(X_valid)
    
    
    # initilaize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    #fit model on training data
    model.fit(X_train, df_train.target.values)
    
    # predict on validation data
    # we need the probablity values as we are calculating auc
    # we will use the probablity of 1s
    
    valid_preds = model.predict_proba(X_valid)[:,1]
    
    # get roc auc curve
    auc = metrics.roc_auc_score(df_valid.target.values,valid_preds)
    
    # print(auc)
    print(f"Fold = {fold}, AUC = {auc}")
    
    

if __name__ == "__main__":
    # run function for fold 0
    for fold_ in range(5):
        run(fold = fold_)

