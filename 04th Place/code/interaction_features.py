import pandas as pd
import numpy as np

def feature_interaction(train,test,labels_tr,labels_te):
    train = pd.concat([train,test]).reset_index(drop=True)
    labels_tr["is_train"]=1
    labels_te["is_train"]=0
    labels = pd.concat([labels_tr,labels_te]).reset_index(drop=True)
    
    phase_list = np.setdiff1d(train.phase.unique().tolist(),['final_rinse'])
    train = train[train.phase.isin(phase_list)]
    
    print("Create Numeric feats")
    real_val_feats = ['supply_flow', 'supply_pressure', 'return_temperature',
           'return_conductivity', 'return_turbidity', 'return_flow']
    
    pick_cols =[]
    for k in range(len(real_val_feats)-1):
        for h in real_val_feats[k+1:]:
            col = real_val_feats[k]+"_"+h
            train[col]= train[real_val_feats[k]]*train[h]
            train.loc[train[col]<0,col]=0
            pick_cols.append(col)
            
    
    new_data = train[pick_cols+["process_id",'phase']]
    
    new_data = new_data.groupby(["process_id","phase"]).sum().reset_index()
    
    final_train_set = labels[["process_id"]]
    
    def create_data(data,col_name,f_data):
        for k in phase_list:
            zz = data[data.phase==k][["process_id",col_name]]
            zz[col_name]= np.log(zz[col_name]+1)
            zz.columns = ["process_id",col_name+" "+k]
            f_data = f_data.merge(zz,on="process_id",how="left")
        return f_data
    
    for cols in pick_cols:
        print(cols)
        final_train_set = create_data(new_data,cols,final_train_set)
    return final_train_set
    
