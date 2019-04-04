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

pd.set_option('display.max_columns', 40)
threshold = 290000


def skip_rows(index):
    # return index %100 > 0
    return False  # load all


def create_xgb_model(params):
    X, y, saveTo, n_estimators = params
    model = xgb.XGBRegressor(objective="reg:linear",
                             learning_rate=0.05,
                             min_child_weight=10,
                             subsample=0.8,
                             colsample_bytree=0.8,
                             silent=0,
                             max_depth=10,
                             n_jobs=4,
                             n_estimators=n_estimators)
    print(model)
    model.fit(X, y)
    print('saving model......')
    dump(model, open(saveTo, "wb"))
    return model



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
tunning_info_path = arguments['--tunning_info_path']
output_directory = arguments['--output_directory']

print('reshaped_training_data_path, output_directory: ', reshaped_training_data_path, output_directory)

final_rinse_total_turbidity_liter = 'final_rinse_total_turbidity_liter'


# for training our model
print("Loading data...")
train_values_phases = pd.read_csv(reshaped_training_data_path, skiprows=skip_rows)
print(train_values_phases.shape)
train_values_phases = train_values_phases.drop(['phases_names'], axis=1)

recipe_metadata_dict = pd.read_csv(recipe_metadata_path).set_index(
    'process_id').to_dict('index')

processes_ids = train_values_phases['process_id']
train_values_phases['pre_rinse'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['pre_rinse'])
train_values_phases['caustic'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['caustic'])
train_values_phases['intermediate_rinse'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['intermediate_rinse'])
train_values_phases['acid'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['acid'])

train_values_phases = train_values_phases.drop(['process_id'], axis=1)

train_values_phases['pipeline'] = pd.Categorical(
    train_values_phases['pipeline'])
dfDummies = pd.get_dummies(train_values_phases['pipeline'], prefix='pipeline')
train_values_phases = pd.concat([train_values_phases, dfDummies], axis=1)
train_values_phases = train_values_phases.drop(['pipeline'], axis=1)

models_tunning_info = None
with open(tunning_info_path) as f:
    models_tunning_info = json.load(f)


# train_values_phases = train_values_phases[:round(
#     train_values_phases.shape[0]*.8)]

print(train_values_phases.shape)
target_squared_inverse_sums = {}
phases_train_values = train_values_phases.groupby(['phases_code'])
for phase_train_values in phases_train_values:
    phases_code = str(phase_train_values[0])
    # if(phases_code != '1'):
    #     continue
    train_values_x = phase_train_values[1]
    train_values_lower = train_values_x[train_values_x[final_rinse_total_turbidity_liter] < threshold]
    train_values_x = [(0, train_values_lower), (1, train_values_x)]
    for item in train_values_x:
        train_values = item[1]
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
            target_squared_inverse_sums[phases_code] = target_squared_inverse_sum
            print(target_squared_inverse_sum)
    
        train_values = train_values.drop(
        ['phases_code', final_rinse_total_turbidity_liter], axis=1)
        
        model_code = None

        if(item[0] == 1):  # full
            model_code = phases_code
        else:
            model_code = phases_code+"_lower"

        model_phaseFile = join(output_directory, 'xgb-model-phase-'+model_code+'.dat')
        model_phase = create_xgb_model(
            (train_values, target, model_phaseFile, int(50 + models_tunning_info[model_code]['estimators'])))


print(target_squared_inverse_sums)
with open(join(output_directory, 'target_squared_inverse_sums.json'), 'w') as file:
     file.write(json.dumps(target_squared_inverse_sums)) # use `json.loads` to do the reverse

print("Finished training")
print("--- %s seconds ---" % ((time() - start_time)))
