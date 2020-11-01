# lbl_xgb.py
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:55:47 2020

@author: aagrawal
"""

import config

import pandas as pd
import xgboost as xgb

from sklearn import metrics, preprocessing


def run(fold):
    
    # load full training data with folds
    df = pd.read_csv(config.TRAIN_FOLDS_FILE)
    
    # all columns are features except id, target and kfold columns
    features = [x for x in df.columns if x not in ("id", "target", "kfold")]
    
    
       
    for col in features:
        # fill all NaN values to "NONE" and convert the columns values to string.
        # It does not matter since all features are categorical so we can convert them to string 
        df.loc[:, col] = df.loc[:, col].astype(str).fillna("NONE") 
        
        # initialize label encoder for each column
        lbl = preprocessing.LabelEncoder()
        
        # fit lable encoder on all data
        lbl.fit(df[col])
        
        # transform all the data
        df.loc[:,col] = lbl.transform(df[col])
    
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop= True) 
    
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True) 
    
    # get training data
    X_train = df_train[features].values
    
    # get validation data
    X_valid = df_valid[features].values
    
        
    # initilaize XGBoost model
    model = xgb.XGBClassifier(n_jobs = -1, max_depth = 7, n_estimators = 200)
    
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
