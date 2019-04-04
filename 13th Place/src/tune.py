# %% import libraries and set input parameters
from time import time
start_time = time()

import json
from os import listdir, path
from os.path import isfile, join, dirname, realpath
from pandas import DataFrame
import xgboost as xgb
from pickle import load, dump
from sys import argv
from multiprocessing import cpu_count, Pool
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 100)
threshold = 290000

def skip_rows(index):
    # return index % 100 > 0
    return False  # load all



def variant_mape(preds, dtrain, target_squared_inverse_sum, is_full):
    # print(dtrain[0],preds[0])
    labels = dtrain
    if(is_full == 1):
        labels = np.array([(1/((value**2) * target_squared_inverse_sum))
                           for value in labels])
        preds = np.array([(1/((value**2) * target_squared_inverse_sum))
                          for value in preds])
        return np.mean(np.abs((labels - preds) / np.maximum(labels, threshold)))
    else:
        labels = np.array([value**2 for value in labels])
        preds = np.array([value**2 for value in preds])
    return np.mean(np.abs((labels - preds) / threshold))


def cvtest(i, params, target_squared_inverse_sum, is_full):
    params["silent"] = 1
    params["n_jobs"] = 4

    plst = list(params.items())

    num_rounds = 1000

    # pass the indexes to your training and validation data
    xgtrain = xgb.DMatrix(train_values[cv[i][0]], label=target[cv[i][0]])
    val_target = target[cv[i][1]]
    val_train = train_values[cv[i][1]]
    xgval = xgb.DMatrix(val_train, label=val_target)

    # define a watch list to observe the change in error f your training and holdout data
    watchlist = [(xgval, 'eval')]

    model = xgb.train(plst,
                      xgtrain,
                      num_rounds,
                      watchlist,
                      early_stopping_rounds=50)   # stops 50 iterations after marginal improvements or drop in performance on your hold out set

    pred_train = model.predict(xgval)
    print('best iteration:', model.best_iteration)
    print('best score:', model.best_score)

    fold_variant_mape = variant_mape(
        pred_train, val_target, target_squared_inverse_sum, is_full)
    print("score for cv", i, "=", fold_variant_mape)
    return(fold_variant_mape, model.best_iteration, cv[i][1], pred_train)



print('args', argv)
arguments_names = []
arguments_values = []

for i in range(1, len(argv)):
    if(argv[i].startswith('--')):
        arguments_names.append(argv[i].lower())
    else:
        arguments_values.append(argv[i])
if(len(arguments_names) != len(arguments_values)):
    print('Number of input paramters is not matched with number of provided values')
    exit()

arguments = {}
for i in range(len(arguments_names)):
    arguments[arguments_names[i]] = arguments_values[i]


reshaped_training_data_path = arguments['--reshaped_training_data_path']
recipe_metadata_path = arguments['--recipe_metadata_path']
output_directory = arguments['--output_directory']

print('reshaped_training_data_path, output_directory: ', reshaped_training_data_path, output_directory)

final_rinse_total_turbidity_liter = 'final_rinse_total_turbidity_liter'


# for training our model
print("Loading data...")
train_values_all = pd.read_csv(reshaped_training_data_path, skiprows=skip_rows)
train_values_all = train_values_all.drop(['phases_names'], axis=1)

# train_values_all = train_values_all[:int(train_values_all.shape[0]*.5)]

recipe_metadata_dict = pd.read_csv(recipe_metadata_path).set_index(
    'process_id').to_dict('index')

processes_ids = train_values_all['process_id']
train_values_all['pre_rinse'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['pre_rinse'])
train_values_all['caustic'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['caustic'])
train_values_all['intermediate_rinse'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['intermediate_rinse'])
train_values_all['acid'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['acid'])


train_values_all['pipeline'] = pd.Categorical(train_values_all['pipeline'])
dfDummies = pd.get_dummies(train_values_all['pipeline'], prefix='pipeline')
train_values_all = pd.concat([train_values_all, dfDummies], axis=1)
train_values_all = train_values_all.drop(['pipeline'], axis=1)

print(train_values_all.shape)

print("Preprocessing data...")
final_infos = {}
phases_train_values = train_values_all.groupby(['phases_code'])
for phase_train_values in phases_train_values:
    phases_code = str(phase_train_values[0])
    # if(phases_code != '1'):
    #     continue
    train_values_x = phase_train_values[1]
    train_values_lower = train_values_x[train_values_x[final_rinse_total_turbidity_liter] < threshold]
    train_values_x = [(0, train_values_lower), (1, train_values_x)]
    for item in train_values_x:
        train_values = item[1]
        print(phases_code, train_values.shape)
        target_squared_inverse_sum = 1
        target = train_values[final_rinse_total_turbidity_liter].values
        if(item[0] == 0):  # lower
            target = np.sqrt(target)
        else:
            target_squared_inverse = 1/(target**2)
            target_squared_inverse_sum = np.sum(target_squared_inverse)
            target = target * (target_squared_inverse /
                               target_squared_inverse_sum)
            target = np.sqrt(target)

        processes_ids = train_values['process_id'].values
        train_values = train_values.drop(
            ['process_id', 'phases_code', final_rinse_total_turbidity_liter], axis=1)
        train_values = np.array(train_values)

        seed = np.random.RandomState(2019)
        from sklearn.model_selection import KFold
        # number of folds
        k = 5
        step_size = 1
        k_steps = int(k/step_size)
        kf = KFold(n_splits=k, shuffle=False)
        cv = kf.split(train_values)
        cv = list(cv)
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.05
        params["min_child_weight"] = 10
        params["subsample"] = 0.8
        params["colsample_bytree"] = 0.8
        params["scale_pos_weight"] = 1.0
        params["max_depth"] = 10
        results = np.repeat(0.0, k_steps)
        trees = np.repeat(0.0, k_steps)

        # processes_predictions = []
        for i in range(k_steps):
            results[i], trees[i], cv_indices, pred_train = cvtest(
                i*step_size, params, target_squared_inverse_sum, item[0])

        print(phases_code,  item[0], 'mean score: ', np.mean(results))
        print(phases_code,  item[0], 'mean trees: ', np.mean(trees))
        if(item[0] == 1):
            final_infos[phases_code] = {'score': np.mean(
                results), 'estimators': np.mean(trees), "count": item[1].shape[0]}
        else:
            final_infos[phases_code+'_lower'] = {'score': np.mean(
                results), 'estimators': np.mean(trees), "count": item[1].shape[0]}


print(final_infos)

with open(join(output_directory, 'tunning_info.json'), 'w') as file:
     file.write(json.dumps(final_infos)) # use `json.loads` to do the reverse
print("Finished training")
print("--- %s seconds ---" % ((time() - start_time)))
