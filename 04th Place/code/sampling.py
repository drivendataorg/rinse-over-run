# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:31:24 2019

@author: vt
"""
import pandas as pd
import numpy as np

def sample_data(train_values,test,train_labels):
    #phase distribution
    test_dist = test[["phase","process_id"]].drop_duplicates()
    total_test = test["process_id"].nunique()
    
    def process_(x):
        z= list(x.phase)
        return "_".join(z)
    
    phase_counts = test_dist.groupby("process_id").apply(lambda x: process_(x))
    
    train_dist = train_values[["phase","process_id"]].drop_duplicates()
    train_dist = train_dist[train_dist.phase!="final_rinse"]
    train_counts = train_dist.groupby("process_id").apply(lambda x: process_(x))
    
    
    #sampling in training set
    all_ids = train_counts[train_counts=="pre_rinse_caustic_intermediate_rinse_acid"].index
    np.random.seed(100)
    pre_rinse_caustic = np.random.choice(all_ids,1000,replace=False)
    all_= np.setdiff1d(all_ids,pre_rinse_caustic)
    
    np.random.seed(200)
    pre_rinse_caustic_intermediate_rinse_acid = np.random.choice(all_,1135,replace=False)
    all_= np.setdiff1d(all_,pre_rinse_caustic_intermediate_rinse_acid)
    
    np.random.seed(300)
    pre_rinse_caustic_intermediate_rinse = np.random.choice(all_,1135,replace=False)
    all_= np.setdiff1d(all_,pre_rinse_caustic_intermediate_rinse)
    prerinse =  all_
    
    other_datast = train_values[train_values.process_id.isin(all_ids)]
    d1 = other_datast[(other_datast.process_id.isin(pre_rinse_caustic))&(other_datast.phase.isin(["pre_rinse","caustic"]))]
    d2 = other_datast[(other_datast.process_id.isin(pre_rinse_caustic_intermediate_rinse_acid))&(other_datast.phase.isin(["pre_rinse","caustic","intermediate_rinse","acid"]))]
    d3 = other_datast[(other_datast.process_id.isin(pre_rinse_caustic_intermediate_rinse))&(other_datast.phase.isin(["pre_rinse","caustic","intermediate_rinse"]))]
    d4 = other_datast[(other_datast.process_id.isin(prerinse))&(other_datast.phase.isin(["pre_rinse"]))]
    
    dataset_1 = pd.concat([d1,d2,d3,d4])
    final_dataset  = pd.concat([train_values[~(train_values.process_id.isin(all_ids))],dataset_1])
    return final_dataset










