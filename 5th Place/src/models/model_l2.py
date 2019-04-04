import pickle
import numpy as np
import pandas as pd
from src.features.utils import mape,blend_dict,target_col,mapeu,to_log,from_log
from src.config import PERSISTED_MODEL_PATH,SUBMISSION_PATH
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import weighted



def train_l2(datasets):
    a=0
    with open(PERSISTED_MODEL_PATH+'model_l1_1.pickle', 'rb') as handle:
        folds=pickle.load(handle)
    with open(PERSISTED_MODEL_PATH+'model_l1_2.pickle', 'rb') as handle:
        folds2=pickle.load(handle)

    with open(PERSISTED_MODEL_PATH+'model_l1_3.pickle', 'rb') as handle:
        folds3=pickle.load(handle)

    blended_folds = blend_dict(folds, folds2, 'x2')
    blended_folds = blend_dict(blended_folds, folds3, 'x3')

    print(blended_folds[0]['models'].keys())
    #generate full datasetl2
    l2_folds=[]
    for id,fold in enumerate(blended_folds):
        full_data_l2=dict()
        full_data_l2['data_l2']=dict()
        x1= str((id+1)%3)
        x2= str((id+2)%3)
        d1=datasets['val_set_'+x1]
        d1=d1.copy()
        d2=datasets['val_set_'+x2]
        d2=d2.copy()
        test=datasets['test_set']
        test=test.copy()
        for mkey,model in fold['models'].items():
            d1[mkey]=model['preds']['val_set_'+x1]
            d2[mkey]=model['preds']['val_set_'+x2]
            test[mkey]=model['preds']['test_set']
        full_data_l2['data_l2']['val_setl2_'+x1]=d1
        full_data_l2['data_l2']['val_setl2_'+x2]=d2
        full_data_l2['data_l2']['test_setl2']=test
        full_data_l2['preds']=dict()
        l2_folds.append(full_data_l2)
    paramsx1 = {
        'objective': 'regression_l1',
        'learning_rate': 0.1,
        'num_leaves': 65,
        'max_depth':4,
        'feature_fraction': 0.20,
        'bagging_fraction': 0.9,
        'boosting_type': 'gbdt',
        'uselog':False,
        #'from':from_sqrt,
        #'to':to_sqrt,
        'usewg':True,
        'train_on':'val_setl2_1',
        'validate_on':'val_setl2_2',

    }
    for i in range(0,1):
        for id_fold,fold in enumerate(l2_folds):
            paramsx1['seed']=i
            paramsx1['num_leaves']=60+2*i
            x1= str((id_fold+1)%3)
            x2= str((id_fold+2)%3)
            paramsx1['train_on']=  'val_setl2_'+x1
            paramsx1['validate_on']=  'val_setl2_'+x2
            pred_dict,y_dict,model_boost=fit4(data_dict=fold['data_l2'],params=paramsx1)
            fold['preds']['ml2_'+str(i)]=pred_dict
            fold['true']=y_dict

    l3_folds = []
    base_cols = ['process_id', 'phase', 'duration', target_col]
    for fold_id, fold in enumerate(l2_folds):
        l3_datasets = {}
        for dataset_key in fold['preds']['ml2_0'].keys():
            cols2keep = [x for x in fold['data_l2'][dataset_key].columns if x in base_cols]
            data = fold['data_l2'][dataset_key][cols2keep].copy()
            for pred_name, pred in fold['preds'].items():
                data[pred_name] = pred[dataset_key]

            data['ml2'] = data[[x for x in fold['preds'].keys()]].median(axis=1)
            if 'val' in dataset_key:
                data = generate_train_cuts(data, dataset_key)
            else:
                data = group_level1_preds(data)
            l3_datasets[dataset_key] = data
        l3_folds.append(l3_datasets)


    # generate l3 validation set and testset
    df_list_val = []
    df_list_test = []
    for id_fold, fold in enumerate(l3_folds):
        df_list_val.append(fold['val_setl2_' + str((id_fold + 2) % 3)])
        test = fold['test_setl2'].copy()
        test.columns = [x + '_' + str(id_fold) for x in test.columns]
        df_list_test.append(test)
    full_val_l3 = pd.concat(df_list_val)
    full_test_l3 = pd.concat(df_list_test, axis=1)

    print(full_val_l3.shape, full_test_l3.shape)

    validations_list = []
    for id_fold, fold in enumerate(l3_folds):
        dataset = fold['val_setl2_' + str((id_fold + 2) % 3)]

        df_list = []
        for col in [x for x in dataset.columns if x.startswith('m')]:
            df = pd.DataFrame({'col': [col], 'score_' + str(id_fold): [mape(dataset[target_col], dataset[col])]})
            df_list.append(df)

        validations_list.append(pd.concat(df_list).set_index('col'))

    df_list = []
    for col in [x for x in full_val_l3.columns if x.startswith('m')]:
        df = pd.DataFrame({'col': [col], 'score': [mape(full_val_l3[target_col], full_val_l3[col])]})
        df_list.append(df)

    validation_stats = pd.concat(df_list)
    validation_stats.sort_values('score')
    col = 'ml2_wgmedianlogdur'
    val_ens = full_val_l3[col].copy()
    val_ens[val_ens < 0] = 1000
    test_ens = full_test_l3[[col + '_0', col + '_1', col + '_2']].median(axis=1)
    print_submissions_stats(val_ens, full_val_l3[target_col], test_ens)

    agg_l2test = l3_folds[0]['test_setl2'].copy()

    agg_l2test['final_rinse_total_turbidity_liter'] = test_ens
    agg_l2test.reset_index(inplace=True)
    agg_l2test[['process_id', 'final_rinse_total_turbidity_liter']].to_csv(SUBMISSION_PATH+'final_submission.csv', index=False)

def generate_train_cuts(data, mode=''):
    eliminations = {'el1': ['final_rinse'],
                    'el2': ['acid', 'final_rinse'],
                    'el3': ['intermediate_rinse', 'acid', 'final_rinse'],
                    'el4': ['caustic', 'intermediate_rinse', 'acid', 'final_rinse'],
                    }

    df_list = []
    process_list = data.process_id.unique()
    for el, list_el in eliminations.items():
        if el == 'el4':
            _, process_el4 = train_test_split(process_list, shuffle=False, test_size=0.35)
            df = data[data['process_id'].isin(process_el4)]
            df = df[~df['phase'].isin(list_el)]
        else:
            df = data[~data['phase'].isin(list_el)]
        print(mode + ' shape for ', el, ':', df.shape)
        df_list.append(group_level1_preds(df))
    agg_df = pd.concat(df_list, ignore_index=True)
    return agg_df



def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def group_level1_preds(filtered_data):
    df_list = []
    model_list = [x for x in filtered_data.columns if x.startswith('m')]
    filtered_data['logdur'] = np.log(1 + 0.1 * filtered_data['duration'] / 10000)



    for col in ['ml2']:
        df = filtered_data.groupby('process_id').apply(lambda x: pd.Series({
            col + '_wgmedianlogdur': weighted.median(x[col], x['logdur'] / x[col]),
        }))
        df_list.append(df)

    if 'final_rinse_total_turbidity_liter' in filtered_data.columns:
        filtered_data['score'] = mapeu(filtered_data['final_rinse_total_turbidity_liter'], filtered_data['ml2'])

        df = filtered_data[filtered_data.groupby('process_id').score.transform('min') == filtered_data['score']]
        df['ml2best'] = df['ml2']
        df = filtered_data.groupby('process_id').apply(lambda x: pd.Series({
            'final_rinse_total_turbidity_liter': x['final_rinse_total_turbidity_liter'].max(),

        }))
        df_list.append(df)

    return pd.concat(df_list, axis=1)


def get_init_score(dataset):
    dataset = dataset.copy()
    return (dataset['m1_wgmedian'] + dataset['m6_wgmedian']) / 2.


def generate_target2(data=None):
    col2keep = ['cat', 'duration',
                'cumulate_diff_flow_avg_cumulate', 'cumulate_target_avg_cumulate',
                'diff_flow_avg_cumulate', 'target_avg_cumulate',
                'return_flow_avg_cumulate', 'return_turbidity_avg_cumulate',
                'return_conductivity_avg_cumulate',
                'return_temperature_avg_cumulate', 'supply_pressure_avg_cumulate',
                'cumulate_target', 'diff_flow', 'target',
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
    col2keep = [x for x in online_data.columns if x.startswith('m')] + col2keep
    online_data = online_data[col2keep]

    return online_data, y


def fit4(data_dict={}, params=None):
    params = params.copy()
    uselog = params.pop('uselog')
    usewg = params.pop('usewg', False)
    train_dataset = params.get('train_on')
    val_dataset = params.get('validate_on')
    if params.get('from') is not None:
        to_trasnform = params.pop('to')
        from_trasnform = params.pop('from')
    elif uselog:
        to_trasnform = to_log
        from_trasnform = from_log

    x_dict = {}
    y_dict = {}
    y_true_dict = {}
    pred_dict = {}
    for dataset_name, dataset in data_dict.items():
        x, y = generate_target2(dataset)
        mono_const = [1 if a.startswith('m') else 0 for a in x.columns]
        print(dataset_name + ' post target gen shape', x.shape)
        y_true_dict.update({dataset_name: y})
        x_dict.update({dataset_name: x})
        if uselog and y is not None:
            y = to_trasnform(y)
        y_dict.update({dataset_name: y})

    # params['mc']= mono_const

    if usewg:
        print('using wg')
        div_train = y_true_dict[train_dataset].copy()
        div_train[div_train < 290000] = 290000
        div_val = y_true_dict[val_dataset].copy()
        div_val[div_val < 290000] = 290000
        d_train = lgbm.Dataset(x_dict[train_dataset], y_dict[train_dataset], weight=1 / div_train,
                               # init_score=get_init_score(x_dict[train_dataset])
                               )
        d_valid = lgbm.Dataset(x_dict[val_dataset], y_dict[val_dataset], weight=1 / div_val,
                               # init_score=get_init_score(x_dict[val_dataset])
                               )
    else:
        d_train = lgbm.Dataset(x_dict[train_dataset], y_dict[train_dataset])
        d_valid = lgbm.Dataset(x_dict[val_dataset], y_dict[val_dataset])

    model = lgbm.train(params, d_train, 50000, valid_sets=[d_train, d_valid], verbose_eval=100,
                       early_stopping_rounds=100,  # feval=mapelgbm
                       )

    for dataset_name, dataset in x_dict.items():
        if dataset_name != train_dataset:
            # ds=lgbm.Dataset(dataset,init_score=get_init_score(dataset))
            pred = model.predict(dataset)  # +get_init_score(dataset)
            if uselog:
                pred = from_log(pred)
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


def print_submissions_stats(val_ensemble, Y_valid, test_ensemble):
    print(test_ensemble.shape)
    print('val mape', mape(Y_valid, val_ensemble))
    print('-------------------------------------------')
    print('val wg median %s', weighted.median(val_ensemble, 1 / val_ensemble))
    print('true val wg median', weighted.median(Y_valid, 1 / Y_valid))
    print('test wg median %s', weighted.median(test_ensemble, 1 / test_ensemble))
    print('-------------------------------------------')
    print('val median', val_ensemble.median())
    print('true val median', Y_valid.median())
    print('test median', test_ensemble.median())
    print('-------------------------------------------')
    print('val min', val_ensemble.min())
    print('true val min', Y_valid.min())
    print('test min', test_ensemble.min())
    print('-------------------------------------------')
