
import pandas as pd
from src.config import RAW_DATA_PATH

from sklearn.model_selection import train_test_split
import numpy as np


import copy


offset=1
target_col = 'final_rinse_total_turbidity_liter'

def get_training_data():
    data = pd.read_csv(RAW_DATA_PATH + 'train_values.csv')
    print('starting_shape %', data.shape)

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    print('end_shape %', data.shape)

    return data

def get_production_data():
    data =  pd.read_csv(RAW_DATA_PATH + 'test_values.csv')
    print('starting_shape %',data.shape)
    data['timestamp']=pd.to_datetime(data['timestamp'])
    print('end_shape %', data.shape)

    return data


def get_datasets(debug=True):
    target_col = 'final_rinse_total_turbidity_liter'
    train = get_training_data()
    train = train[~train['phase'].isin(['final_rinse'])]
    print(train.shape)
    test = get_production_data()
    labels = pd.read_csv(RAW_DATA_PATH + 'train_labels.csv')
    categorical_feat = ['pipeline', 'object_id', 'phase']
    print(train.shape, test.shape)
    train['is_train'] = 1
    test['is_train'] = 0
    full_data = pd.concat([train, test], ignore_index=True)
    for col in categorical_feat:
        full_data[col] = full_data[col].astype('category')

    train = full_data[full_data['is_train'] == 1].drop(['is_train'], axis=1)
    test = full_data[full_data['is_train'] == 0].drop(['is_train'], axis=1)
    print(train.shape, test.shape)
    process_id_list = train.process_id.unique()

    if debug:
        process_id_list, _ = train_test_split(process_id_list, shuffle=False, test_size=0.9900)
        test=test.head(10000)
    process_id_val, process_id_train = train_test_split(process_id_list, shuffle=False, test_size=0.3333)
    process_id_val1, process_id_val2 = train_test_split(process_id_val, shuffle=False, test_size=0.5)

    assert (min(process_id_val) < max(process_id_train))

    datasets = {}
    datasets.update({'val_set_0': train[train['process_id'].isin(process_id_train)]})
    datasets.update({'val_set_1': train[train['process_id'].isin(process_id_val1)]})
    datasets.update({'val_set_2': train[train['process_id'].isin(process_id_val2)]})
    datasets.update({'test_set': test})
    return datasets,labels




def mape(y_true, y_pred):
    div=y_true.copy()
    div[div<290000]=290000
    return np.mean(np.abs((y_true - y_pred) / div))

def mapeu(y_true, y_pred):
    div=y_true.copy()
    div[div<290000]=290000
    return np.abs((y_true - y_pred) / div)

def mapelog( y_pred,y_true):
    y=from_log(y_true.get_label())
    div=y
    div[div<290000]=290000
    return np.mean(np.abs((y - from_log(y_pred)) / div))

def modifify_y(y):
    y_mod=y
    #y_mod[y_mod<29000]=29000
    return y_mod


def to_log(tar):
    tar=tar.copy()
    return np.log(1+tar)

def from_log(tar):
    return (np.exp(tar)-offset)

def to_sqrt(tar):
    tar=tar.copy()
    return np.power(tar,1/2)

def from_sqrt(tar):
    tar.copy()
    tar[tar<0]=0.01
    return np.power(tar,2)





def blend_dict(folds1,folds2,f2_suffix='xx'):
    final_fold=copy.deepcopy(folds1)
    for fold_id,fold in enumerate(final_fold):
        for model_name,model in folds2[fold_id]['models'].items():
            fold['models'][model_name+f2_suffix]=copy.deepcopy(model)

    return final_fold
