import pandas as pd
import numpy as np
def final_rinse_feats(train):
    train = train[train.phase=="final_rinse"]
    train["turb*flow"] = train['return_turbidity']*train['return_flow']
    #train.loc[train["turb*flow"]<0,"turb*flow"] = 0
    
    real_val_feats = ['supply_flow', 'supply_pressure', 'return_temperature',
           'return_conductivity', 'return_turbidity', 'return_flow','tank_level_pre_rinse',
           'tank_level_caustic', 'tank_level_acid', 'tank_level_clean_water',
           'tank_temperature_caustic','tank_temperature_pre_rinse',
           'tank_temperature_acid', 'tank_concentration_caustic',
           'tank_concentration_acid',"turb*flow"]
    
    agg_fns = ["mean","median","std","min","max"]
    #agg_fns = ["median","std","min","max"]
    count=0
    for feat in real_val_feats:
        print(feat)
        zz = train[['object_id',feat]].groupby(['object_id']).agg(agg_fns).reset_index()
        zz.columns = zz.columns.droplevel(0)
        zz["range"] = zz["max"]-zz["min"]
        zz.columns = ['object_id']+[feat+"_"+k for k in agg_fns+["range"]]
        if count==0:
            count+=1
            train_agg = zz
        else:
            train_agg = train_agg.merge(zz,on=['object_id'],how = "left")
    
    
    print("Create Binary Feats")
    binary_feat = ['supply_pump',
           'supply_pre_rinse', 'supply_caustic', 'return_caustic', 'supply_acid',
           'return_acid', 'supply_clean_water', 'return_recovery_water',
           'return_drain', 'object_low_level','tank_lsh_caustic', 
           'tank_lsh_clean_water', 'target_time_period']
    for feat in binary_feat:
        print(feat)
        zz = train[['object_id',feat]].groupby(['object_id']).apply(lambda x: x.sum()/x.size)[feat].reset_index()
        train_agg = train_agg.merge(zz,on=['object_id'],how = "left")
    
    return train_agg