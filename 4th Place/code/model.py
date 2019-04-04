# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:21:31 2019

@author: vt
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import scipy
from sampling import sample_data
from local_feats import local_feats
from final_rinse_feats import final_rinse_feats
from interaction_features import feature_interaction
from global_features import global_features

def build_model(train_file,test_file,label_file,submission_file,metadata_file,submission_path):
    print("reading datasets")
    train_values = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    #read metadata
    metadata = pd.read_csv(metadata_file)
    #train = pd.concat([train,test]).reset_index(drop=True)
    train_labels = pd.read_csv(label_file)
    sub_fil = pd.read_csv(submission_file)
    sub_fil["final_rinse_total_turbidity_liter"] =np.exp(np.log(train_labels["final_rinse_total_turbidity_liter"]).mean())
    print("sampling training dataset")
    data_sample = sample_data(train_values,test,train_labels)
    print("Feature Extraction")
    print("Extract local features")
    dataset_full = local_feats(data_sample,test,train_labels,sub_fil)
    
    metadata.drop("final_rinse",axis=1,inplace=True)
    #metadata["planned_stages_sum"] = metadata[['pre_rinse', 'caustic', 'intermediate_rinse', 'acid']].sum(axis=1)
    metadata.columns = ["process_id"]+['pre_rinse_sch', 'caustic_sch', 'intermediate_rinse_sch', 'acid_sch']
    
    dataset_full = dataset_full.merge(metadata,on="process_id",how="left")
    
    
    print("creating final rinse features")
    features_set4 = final_rinse_feats(train_values)
    dataset_full = dataset_full.merge(features_set4,on=["object_id"],how="left")
    
    print("Creating feature interactions")
    features_set5 = feature_interaction(data_sample,test,train_labels,sub_fil)
    dataset_full = dataset_full.merge(features_set5,on=["process_id"],how="left")
    
    test = dataset_full[dataset_full["is_train"]==0]
    train = dataset_full[dataset_full["is_train"]==1]
    print("Creating global features")
    dataset_full_diff = global_features(data_sample,test,train_labels,sub_fil)
    #create features using response variables
    def groupby_(vals,col_name,train,test):
        #hh = train[['final_rinse_total_turbidity_liter']+vals]
        #hh['final_rinse_total_turbidity_liter'] = np.log(hh['final_rinse_total_turbidity_liter'])
        dd = train[['final_rinse_total_turbidity_liter']+[vals[0]]].groupby(vals[0]).mean().reset_index()
        dd.columns = [vals[0]]+[ col_name]
        dd[col_name] = np.log(dd[col_name])
        #dd['turbidity_median'] =dd['turbidity_median']
        train = train.merge(dd,on=vals[0],how="left")
        test = test.merge(dd,on=vals[0],how="left")
        
        
        dd = train[['final_rinse_total_turbidity_liter']+[vals[1]]].groupby(vals[1]).mean().reset_index()
        dd.columns = [vals[1]]+[ col_name+"_pipeline"]
        dd[col_name+"_pipeline"] = np.log(dd[col_name+"_pipeline"])
        #dd['turbidity_median'] =dd['turbidity_median']
        train = train.merge(dd,on=vals[1],how="left")
        test = test.merge(dd,on=vals[1],how="left")
        
        
    
        ff = train[['final_rinse_total_turbidity_liter']+vals].groupby(vals).apply(lambda x: scipy.stats.iqr(x)).reset_index()
        ff.columns = vals+[ col_name+"_iqr"]
        ff[col_name+"_iqr"] = np.log(ff[col_name+"_iqr"])
        train = train.merge(ff,on=vals,how="left")
        test = test.merge(ff,on=vals,how="left")
        per_25 = train[['final_rinse_total_turbidity_liter']+vals].groupby(vals).apply(lambda x: np.percentile(x,25)).reset_index()
        per_25.columns = vals+[ col_name+"_per_25"]
        per_25[col_name+"_per_25"] = np.log(per_25[col_name+"_per_25"])
        train = train.merge(per_25,on=vals,how="left")
        test = test.merge(per_25,on=vals,how="left")
        per_75 = train[['final_rinse_total_turbidity_liter']+vals].groupby(vals).apply(lambda x: np.percentile(x,75)).reset_index()
        per_75.columns = vals+[ col_name+"_per_75"]
        per_75[col_name+"_per_75"] = np.log(per_75[col_name+"_per_75"])
        train = train.merge(per_75,on=vals,how="left")
        test = test.merge(per_75,on=vals,how="left")
        return train, test
    
    train = train.merge(dataset_full_diff,on=["process_id"],how="left")
    test = test.merge(dataset_full_diff,on=["process_id"],how="left")
    
    
    train["object_0"]=train["object_id"].apply(lambda x: str(x)[0])
    test["object_0"]=test["object_id"].apply(lambda x: str(x)[0])
    
    def conv_int(x):
        try:
            return int(x)
        except:
            return np.nan
    
    train["object_0"] = train["object_0"].apply(lambda x: conv_int(x))
    test["object_0"] = test["object_0"].apply(lambda x: conv_int(x))
    
    train["pipeline"] = train["pipeline"].apply(lambda x: conv_int(x))
    test["pipeline"] = test["pipeline"].apply(lambda x: conv_int(x))
       
    label = 'final_rinse_total_turbidity_liter'
    
    u = train.sum()
    columns = u[u==0].index
    
    not_req = ['process_id',"target_time_period","is_train"]+columns.tolist()
        
    def mape_(actual,pred):
        abs_ = np.abs(actual-pred)
        denom = actual.copy()
        denom[denom<290000]=290000
        f = abs_/denom
        return f.mean()
    
    def mape(actual,pred):
        abs_ = np.abs(actual-pred)
        denom = actual.copy()
        denom[denom<290000]=290000
        f = abs_/denom
        return f.mean()
    
    
    def mape_error(preds, train_data):
        labels = train_data.get_label()
        #print(labels)
        return 'error', mape_(np.exp(labels),np.exp(preds)), False
        #return 'error', mape_(labels,preds), False
    
    train = train[~(train["process_id"].isin([22502,
    20308,
    27008,
    23830,
    26908,
    24186,
    25683,
    27676,
    21941,
    20341,
    23756,
    25376,
    21833,
    20146,
    21637,
    21216]))].reset_index(drop=True)
    
    #train.ix[train[label]>5000000,label]=5000000
    train_median= train[label].median()
    losg=[]
    test_vals = []
    
    folds =KFold(n_splits=10, shuffle=True, random_state=123) 
    ROUNDS = 9000
    params = {
    	'objective': 'Quantile',
        'boosting': 'gbdt',
        'learning_rate': 0.006,
        'verbose': 1,
        'num_leaves': 65,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'bagging_seed': 12345,
        'feature_fraction': 0.6,
        'feature_fraction_seed': 123,
        'max_bin': 150,
        'max_depth':7,
        'num_rounds': ROUNDS,
        'min_sum_hessian_in_leaf':0.25,
        'min_data_in_leaf':20,
        "alpha":0.3,
        "min_data_in_bin":30
    
    }
    
    print("build models")
    for train_cv, test_cv in folds.split(train):
        train_set_1 = train.ix[train_cv,:]
        val_set=train.ix[test_cv,:]
        
        train_set_1,val_set = groupby_(["object_id","pipeline"],"pip_obj_median",train_set_1,val_set)
        obj_counts = train_set_1.object_id.value_counts()
        obj_counts = obj_counts[obj_counts>10].index
        train_set_1.ix[~(train_set_1.object_id.isin(obj_counts)),"pip_obj_median"]= np.log(train_median)
        
        train_set_1.ix[~(train_set_1.object_id.isin(obj_counts)),"pip_obj_median_iqr"]= np.log(train_median)
        train_set_1.ix[~(train_set_1.object_id.isin(obj_counts)),"pip_obj_median_per_75"]= np.log(train_median)
        train_set_1.ix[~(train_set_1.object_id.isin(obj_counts)),"pip_obj_median_per_25"]= np.log(train_median)
        #train_set_1.ix[~(train_set_1.object_id.isin(obj_counts)),"pip_obj_median_min"]= np.log(720014.8917)
        
        #val_set.ix[~(val_set.object_id.isin(obj_counts)),"pip_obj_median"]= np.log(720014.8917)
        req_cols = np.setdiff1d(train_set_1.columns,not_req+[label])
        train_labels_1 =train_set_1[label]
        #train_labels_1.loc[train_labels_1>6000000]=6000000
        train_labels_1 = np.log(train_labels_1)
        train_lgb = lgb.Dataset(train_set_1[req_cols], train_labels_1)
        val_lgb = lgb.Dataset(val_set[req_cols], np.log(val_set[label]))
        model = lgb.train(params, train_lgb,feval = mape_error, num_boost_round=ROUNDS,valid_sets=val_lgb, early_stopping_rounds=400)
        #model = lgb.train(params, train_lgb,feval = mape_error, num_boost_round=ROUNDS)
        pred_val = model.predict(val_set[req_cols])
        test1 = test.copy()
        train_set_1,test1= groupby_(["object_id","pipeline"],"pip_obj_median",train,test1)
        pred = model.predict(test1[req_cols])
        test_vals.append(np.exp(pred).tolist())
        mae =mape(val_set[label],np.exp(pred_val))
        #test_pred.append(model.predict_proba(test_set[train_set_1.columns]))
        print(mae)
        losg.append(mae)
        #process_id+=val_set.process_id.tolist()
        #actual+=val_set[label].tolist()
        #predicted+=list(np.exp(pred_val))
        #object_id+=val_set.object_id.tolist()
        #pipeline+=val_set.pipeline.tolist()
    
    print("mean:",np.mean(losg))
    print("std:",np.std(losg))
    
    
    print("creating submission file") 
    
    test_vals = np.array(test_vals)
    test_vals = pd.DataFrame(test_vals)
    test_vals =test_vals.T.median(axis=1)
    
    submiss = pd.DataFrame()
    submiss["process_id"] = test["process_id"]
    submiss[label]=test_vals
    
    submiss.to_csv(submission_path+"final_submission.csv",index=False)
    

