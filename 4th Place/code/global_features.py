import pandas as pd
import numpy as np

dir_path= "C:\\weird_kaggle\\forecasting\\"
def global_features(train,test,labels_tr,labels_te):
    train = pd.concat([train,test]).reset_index(drop=True)
    labels_tr["is_train"]=1
    labels_te["is_train"]=0
    labels = pd.concat([labels_tr,labels_te]).reset_index(drop=True)
    
    phase_list = np.setdiff1d(train.phase.unique().tolist(),['final_rinse'])
    train = train[train.phase.isin(phase_list)]
    
    print("Create Numeric feats")
    real_val_feats = ['supply_flow', 'supply_pressure', 'return_temperature',
           'return_conductivity', 'return_turbidity', 'return_flow','tank_level_pre_rinse',
           'tank_level_caustic', 'tank_level_acid', 'tank_level_clean_water',
           'tank_temperature_caustic','tank_temperature_pre_rinse',
           'tank_temperature_acid', 'tank_concentration_caustic',
           'tank_concentration_acid']
    
    agg_fns = ["mean","median","min","max"]
    count=0
    for feat in real_val_feats:
        print(feat)
        #train[feat+"_diff"]= train[['process_id','phase',feat]].groupby(['process_id','phase']).diff()
        zz = train[['process_id',feat]].groupby(['process_id']).agg(agg_fns).reset_index()
        zz.columns = zz.columns.droplevel(0)
        #zz["range"] = zz["max"]-zz["min"]
        #zz.columns = ['process_id']+[feat+"_"+k for k in agg_fns+["range"] ]
        zz.columns = ['process_id']+[feat+"_global"+"_"+k for k in agg_fns]
        if count==0:
            count+=1
            train_agg = zz
        else:
            train_agg = train_agg.merge(zz,on=['process_id'],how = "left")
    return train_agg


