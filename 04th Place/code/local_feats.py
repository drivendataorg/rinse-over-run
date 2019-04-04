# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:48:34 2019

@author: vt
"""

import pandas as pd
import numpy as np

dir_path= "C:\\weird_kaggle\\forecasting\\"

def local_feats(train,test,labels_tr,labels_te):
    train = pd.concat([train,test]).reset_index(drop=True)
    labels_tr["is_train"]=1
    labels_te["is_train"]=0
    labels = pd.concat([labels_tr,labels_te]).reset_index(drop=True)
    
    phase_list = np.setdiff1d(train.phase.unique().tolist(),['final_rinse'])
    train = train[train.phase.isin(phase_list)]
    train["turb*flow"] = train['return_turbidity']*train['return_flow']
    #train.loc[train["turb*flow"]<0,"turb*flow"] = 0
    #train["sup/returnflow"] = train['supply_flow']/train['return_flow']
    
    print("Create Numeric feats")
    real_val_feats = ['supply_flow', 'supply_pressure', 'return_temperature',
           'return_conductivity', 'return_turbidity', 'return_flow','tank_level_pre_rinse',
           'tank_level_caustic', 'tank_level_acid', 'tank_level_clean_water',
           'tank_temperature_caustic','tank_temperature_pre_rinse',
           'tank_temperature_acid', 'tank_concentration_caustic',
           'tank_concentration_acid',"turb*flow"]
    
    agg_fns = ["mean","median","min","max"]
    #agg_fns = ["median","std","min","max"]
    count=0
    for feat in real_val_feats:
        print(feat)
        zz = train[['process_id','phase',feat]].groupby(['process_id','phase']).agg(agg_fns).reset_index()
        zz.columns = zz.columns.droplevel(0)
       # zz["range"] = zz["max"]-zz["min"]
        zz.columns = ['process_id','phase']+[feat+"_"+k for k in agg_fns]
        if count==0:
            count+=1
            train_agg = zz
        else:
            train_agg = train_agg.merge(zz,on=['process_id','phase'],how = "left")
    
    
    print("Create Binary Feats")
    binary_feat = ['supply_pump',
           'supply_pre_rinse', 'supply_caustic', 'return_caustic', 'supply_acid',
           'return_acid', 'supply_clean_water', 'return_recovery_water',
           'return_drain', 'object_low_level','tank_lsh_caustic', 
           'tank_lsh_clean_water', 'target_time_period']
    for feat in binary_feat:
        print(feat)
        zz = train[['process_id','phase',feat]].groupby(['process_id','phase']).apply(lambda x: x.sum()/x.size).reset_index()
        train_agg = train_agg.merge(zz,on=['process_id','phase'],how = "left")
    
    #pipeline feat   
    pipeline = train[['process_id',"pipeline"]].groupby(['process_id',"pipeline"]).size().reset_index()
    pipeline.columns = ['process_id',"pipeline"]+["sample_count"]
    pipeline["pipeline"]=pipeline["pipeline"].apply(lambda x: x[1:])
    
    #object_id feats
    objective = train[['process_id',"object_id"]].groupby(['process_id',"object_id"]).size().reset_index()
    objective = objective[['process_id',"object_id"]]
    
    #create date feats
    train["timestamp"] =  pd.to_datetime(train["timestamp"])
    train["day"] = train["timestamp"].dt.day
    #train["month"] = train["timestamp"].dt.month
    train["time"] = train["timestamp"].dt.minute
    train_date = train[['process_id','phase',"timestamp"]].groupby(['process_id','phase']).apply(lambda x: x.max()-x.min()).reset_index()
    train_date["timestamp"] = train_date["timestamp"]/10**9
    
    train_time = train[['process_id','phase',"day","time"]].groupby(['process_id','phase']).mean().reset_index()
    
    #create the final_dataset
    
    final_train_set = labels.merge(pipeline,on="process_id",how="left")
    final_train_set = final_train_set.merge(objective,on="process_id",how="left")
    
    def create_data(data,col_name,f_data):
        for k in phase_list:
            zz = data[data.phase==k][["process_id",col_name]]
            zz.columns = ["process_id",col_name+" "+k]
            f_data = f_data.merge(zz,on="process_id",how="left")
        return f_data
    
    final_train_set = create_data(train_date,"timestamp",final_train_set)
    
    for cl in ["day","time"]:
        final_train_set = create_data(train_time,cl,final_train_set)
    
    
    need_cols = np.setdiff1d(train_agg.columns,['process_id', 'phase'])
    for cols in need_cols:
        print(cols)
        final_train_set = create_data(train_agg,cols,final_train_set)
    return final_train_set












    

