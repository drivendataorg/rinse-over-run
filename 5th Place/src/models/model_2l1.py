import numpy as np
import pandas as pd
from src.config import RAW_DATA_PATH,PERSISTED_MODEL_PATH
import weighted
from src.features.utils import mape, to_log, from_log, from_sqrt, to_sqrt, get_datasets
import pickle
import lightgbm as lgbm



def create_new_cols(data):
    data = data.assign(target=np.maximum(data.return_flow, 0) * data.return_turbidity)
    data['diff_flow'] = data['supply_flow'] - data['return_flow']

    data['cumulate_target'] = data.groupby('process_id')['target'].cumsum()
    data['cumulate_diff_flow'] = data.groupby('process_id')['diff_flow'].cumsum()

    for col in data.dtypes[data.dtypes == 'float64'].index:
        print(col)
        data[col + '_avg_cumulate'] = np.array(data.groupby('process_id')[col].expanding().mean())

    data['duration'] = pd.to_numeric(
        data.groupby('process_id').apply(lambda x: (x['timestamp'] - x['timestamp'].min()))).values / 200000

    data['tank_diff_pre_rinse'] = np.array(data.groupby('process_id', as_index=False).apply(
        lambda x: x['tank_level_pre_rinse'] - x['tank_level_pre_rinse'].head(1).values[0])) / data['duration']
    data['tank_diff_caustic'] = np.array(data.groupby('process_id', as_index=False).apply(
        lambda x: x['tank_level_caustic'] - x['tank_level_caustic'].head(1).values[0])) / data['duration']
    data['tank_diff_acid'] = np.array(data.groupby('process_id', as_index=False).apply(
        lambda x: x['tank_level_acid'] - x['tank_level_acid'].head(1).values[0])) / data['duration']
    data['tank_diff_water'] = np.array(data.groupby('process_id', as_index=False).apply(
        lambda x: x['tank_level_clean_water'] - x['tank_level_clean_water'].head(1).values[0])) / data['duration']

    data['hour'] = data['timestamp'].dt.hour
    data['dayofweek'] = data['timestamp'].dt.dayofweek
    return data


def pipeline(data):
    data = create_new_cols(data)
    meta = pd.read_csv(RAW_DATA_PATH + 'recipe_metadata.csv')
    meta['cat'] = meta['pre_rinse'] + 2 * meta['caustic'] + 4 * meta['intermediate_rinse'] + 8 * meta['acid'] + 16 * \
                  meta['final_rinse']
    meta['cat'] = meta['cat'].astype('category')
    data = pd.merge(data, meta, on='process_id')
    return data

def train_model_2(datasets,labels):
    for dataset_name, dataset in datasets.items():
        print(dataset_name + ' start shape: ' + str(dataset.shape))
        dataset = pipeline(dataset)
        datasets.update({dataset_name: dataset})
        print(dataset_name + ' post pipeline shape: ' + str(dataset.shape))

    categorical_feat = ['pre_rinse', 'caustic', 'intermediate_rinse', 'acid', 'final_rinse']
    keep_columns = [col for col in datasets['val_set_1'].columns if len(datasets['val_set_1'][col].unique()) > 1]
    for dataset_name, dataset in datasets.items():
        dataset[categorical_feat] = dataset[categorical_feat].astype('category')
        dataset = dataset[keep_columns]
        datasets.update({dataset_name: dataset})
        print(dataset_name + ' end shape: ' + str(dataset.shape))

    for dataset_name, dataset in datasets.items():
        if dataset_name != 'test_set':
            dataset = pd.merge(dataset, labels, on='process_id')
            datasets.update({dataset_name: dataset})

    params1 = {
        'objective': 'regression_l1',
        'learning_rate': .1,
        'num_leaves': 154,
        'max_depth': 8,
        'feature_fraction': 0.60,
        'bagging_fraction': 0.5,
        'max_delta_step': 0.5,
        'boosting_type': 'goss',
        'uselog': False,
        'usewg': True,

    }
    params2 = {
        'objective': 'mape',
        'learning_rate': 0.01,
        'num_leaves': 184,
        'max_depth': 12,
        'feature_fraction': 0.60,
        'bagging_fraction': 0.8,
        'boosting_type': 'goss',
        'uselog': False,

    }
    params3 = {
        'objective': 'regression_l1',
        'learning_rate': 0.05,
        'num_leaves': 69,
        'max_depth': 7,
        'feature_fraction': 0.70,
        'bagging_fraction': 0.8,
        'boosting_type': 'gbdt',
        'uselog': True,

    }
    params4 = {
        'objective': 'huber',
        'alpha': 0.1,
        'learning_rate': 100000.5,
        'num_leaves': 169,
        'max_depth': 12,
        'feature_fraction': 0.70,
        'bagging_fraction': 0.8,
        'boosting_type': 'gbdt',
        'uselog': False,
        'usewg': True,

    }
    params5 = {
        'objective': 'huber',
        'alpha': 0.1,
        'learning_rate': 100000.5,
        'num_leaves': 169,
        'max_depth': 12,
        'feature_fraction': 0.70,
        'bagging_fraction': 0.8,
        'boosting_type': 'goss',
        'uselog': False,
        'usewg': True,

    }
    params6 = {
        'objective': 'regression_l2',
        'learning_rate': 1.1,
        'num_leaves': 144,
        'max_depth': 9,
        'feature_fraction': 0.40,
        'bagging_fraction': 0.5,
        'max_delta_step': 0.5,
        'boosting_type': 'gbdt',
        'uselog': True,
        'from': from_sqrt,
        'to': to_sqrt,
        'usewg': True,

    }
    params7 = {
        'objective': 'regression_l1',
        'learning_rate': .01,
        'num_leaves': 144,
        'max_depth': 9,
        'feature_fraction': 0.40,
        'bagging_fraction': 0.5,
        'max_delta_step': 0.5,
        'boosting_type': 'gbdt',
        'uselog': False,
        'usewg': True,

    }
    params8 = {
        'objective': 'regression_l1',
        'learning_rate': .01,
        'num_leaves': 94,
        'max_depth': 8,
        'feature_fraction': 0.60,
        'bagging_fraction': 0.2,
        'max_delta_step': 1.5,
        'boosting_type': 'gbdt',
        'uselog': False,
        'usewg': True,

    }
    params9 = {
        'objective': 'regression_l1',
        'learning_rate': .1,
        'num_leaves': 154,
        'max_depth': 8,
        'boosting_type': 'dart',
        'uselog': False,
        'usewg': True,
    }
    offset = 1
    uselog = False

    models_l1 = {
        'm1': {'param': params1},
        'm2': {'param': params2},
        'm3': {'param': params3},
        'm4': {'param': params4},
        'm5': {'param': params5},
        'm6': {'param': params6},
        'm7': {'param': params7},
        'm8': {'param': params8},

        'm9': {'param': params9},

    }

    # pred_val1,pred_test1,Y_valid,model=fit2(X_train=agg_train_with_tar,X_valid=agg_val_with_tar,X_test=test_set,params=params1,uselog=True)
    folds = [dict(), dict(), dict()]
    end_fold = []
    import copy

    for key_fold, fold in enumerate(folds):
        tutto_qui = dict()
        train_on = str('val_set_' + str(key_fold))
        validate_on = 'val_set_' + str((key_fold + 1) % len(folds))
        print(key_fold, train_on, validate_on)
        tutto_qui['models'] = copy.deepcopy(models_l1)
        for key_model, model in tutto_qui['models'].items():
            print('training ', key_model)
            model['param'].update({'train_on': train_on})
            model['param'].update({'validate_on': validate_on})

            pred_dict, y_dict, model_boost = fit3(data_dict=datasets, params=model['param'].copy())
            model['preds'] = pred_dict.copy()
            print(pred_dict.keys())
            model['true'] = y_dict.copy()
            print(y_dict.keys())
            model['booster'] = model_boost
        end_fold.append(tutto_qui)

        folds = copy.deepcopy(end_fold)
        with open(PERSISTED_MODEL_PATH+'model_l1_2.pickle', 'wb') as handle:
            pickle.dump(folds, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_target(data=None):
    col2keep = ['cat', 'tank_diff_acid',
                'tank_diff_pre_rinse', 'duration',
                'cumulate_diff_flow_avg_cumulate', 'cumulate_target_avg_cumulate',
                'diff_flow_avg_cumulate', 'target_avg_cumulate',
                'return_flow_avg_cumulate', 'return_turbidity_avg_cumulate',
                'return_conductivity_avg_cumulate',
                'return_temperature_avg_cumulate', 'supply_pressure_avg_cumulate',
                'supply_flow_avg_cumulate', 'cumulate_diff_flow',
                'cumulate_target', 'diff_flow', 'target', 'supply_pump',
                'return_flow', 'return_turbidity', 'return_conductivity',
                'return_temperature', 'supply_pressure', 'supply_flow', 'pipeline',
                'object_id']

    col = 'final_rinse_total_turbidity_liter'
    if col in data.columns:
        y = data[col].copy()
    else:
        y = None
    online_data = data[[x for x in data.columns if x not in ['row_id', 'timestamp', 'flow', 'process_id', 'turbidity',
                                                             'final_rinse_total_turbidity_liter']]].copy()
    online_data = online_data[col2keep]

    return online_data, y


def fit3(data_dict={}, params=None):
    params = params.copy()
    uselog = params.pop('uselog')
    usewg = params.pop('usewg', False)
    train_dataset = params.get('train_on')
    val_dataset = params.get('validate_on')
    print('training on', train_dataset)
    print('validate on', val_dataset)
    if params.get('from') is not None:
        to_trasnform = params.pop('to')
        from_trasnform = params.pop('from')
    elif uselog:
        to_trasnform = to_log
        from_trasnform = from_log

    x_dict = dict()
    y_dict = dict()
    y_true_dict = dict()
    pred_dict = dict()
    for dataset_name, dataset in data_dict.items():
        x, y = generate_target(dataset)
        print(dataset_name + ' post target gen shape', x.shape)
        y_true_dict.update({dataset_name: y})
        x_dict.update({dataset_name: x})
        if uselog and y is not None:
            y = to_trasnform(y)
        y_dict.update({dataset_name: y})

    if usewg:
        print('using wg')
        # take the labels
        div_train = y_true_dict[train_dataset].copy()
        # set labels<290000 to 290000
        div_train[div_train < 290000] = 290000
        div_val = y_true_dict[val_dataset].copy()
        div_val[div_val < 290000] = 290000
        #create ligthgbm vector with weigts
        d_train = lgbm.Dataset(x_dict[train_dataset], y_dict[train_dataset], weight=1 / div_train)
        d_valid = lgbm.Dataset(x_dict[val_dataset], y_dict[val_dataset], weight=1 / div_val)
    else:
        d_train = lgbm.Dataset(x_dict[train_dataset], y_dict[train_dataset])
        d_valid = lgbm.Dataset(x_dict[val_dataset], y_dict[val_dataset])

    model = lgbm.train(params, d_train, 5000, valid_sets=[d_train, d_valid], verbose_eval=100,
                       early_stopping_rounds=100)

    for dataset_name, dataset in x_dict.items():
        if dataset_name != train_dataset:
            pred = model.predict(dataset)
            if uselog:
                pred = from_trasnform(pred)
            pred_dict.update({dataset_name: pred})

    print('mape validation score %s', mape(y_true_dict[val_dataset], pred_dict[val_dataset]))
    print('val wg median %s', weighted.median(pred_dict[val_dataset], 1 / pred_dict[val_dataset]))

    for dataset_name, dataset in x_dict.items():
        if dataset_name != train_dataset:
            if y_dict[dataset_name] is not None:
                print(dataset_name + 'mape  score ' + str(mape(y_true_dict[dataset_name], pred_dict[dataset_name])))
            print(dataset_name + 'val wg median and median',
                  weighted.median(pred_dict[dataset_name], 1 / pred_dict[dataset_name]),
                  np.median(pred_dict[dataset_name]))

    return pred_dict, y_true_dict, model


def main():
    datsets,labels=get_datasets()
    train_model_2(datsets,labels)

if __name__=='__main__':
    debug = True
    main()