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

threshold = 290000


def scale_value(value):
    if(value < threshold):
        return value*.962
    else:
        return value*.931


def skip_rows(index):
    # return index % 100 > 0
    return False





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


reshaped_test_data_path = arguments['--reshaped_test_data_path']
recipe_metadata_path = arguments['--recipe_metadata_path']
models_directory = arguments['--models_directory']
target_squared_inverse_sums_path = arguments['--target_squared_inverse_sums_path']
output_directory = arguments['--output_directory']

print('reshaped_test_data_path, recipe_metadata_path, models_directory, target_squared_inverse_sums_path, output_directory: ', reshaped_test_data_path, recipe_metadata_path, models_directory, target_squared_inverse_sums_path, output_directory)


target_squared_inverse_sums = None
with open(target_squared_inverse_sums_path) as f:
    target_squared_inverse_sums = json.load(f)

print('Loading data...')
inputData_phases = pd.read_csv(reshaped_test_data_path, skiprows=skip_rows)
print(inputData_phases.shape)

# inputData_phases = inputData_phases[int(inputData_phases.shape[0]*.8):]


recipe_metadata_dict = pd.read_csv(recipe_metadata_path).set_index(
    'process_id').to_dict('index')


print('preprocessing....')
# drop columns that are not valued
final_rinse_total_turbidity_liter= 'final_rinse_total_turbidity_liter'
if(final_rinse_total_turbidity_liter in inputData_phases.columns):
    inputData_phases = inputData_phases.drop(
        [final_rinse_total_turbidity_liter], axis=1)

processes_ids = inputData_phases['process_id']
inputData_phases['pre_rinse'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['pre_rinse'])
inputData_phases['caustic'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['caustic'])
inputData_phases['intermediate_rinse'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['intermediate_rinse'])
inputData_phases['acid'] = processes_ids.map(
    lambda process_id: recipe_metadata_dict[process_id]['acid'])

inputData_phases['pipeline'] = pd.Categorical(inputData_phases['pipeline'])
dfDummies = pd.get_dummies(inputData_phases['pipeline'], prefix='pipeline')
inputData_phases = pd.concat([inputData_phases, dfDummies], axis=1)
inputData_phases = inputData_phases.drop(['pipeline'], axis=1)


model_phase_features = ['count', 'supply_flow_mean', 'supply_flow_std', 'supply_flow_min', 'supply_flow_25%', 'supply_flow_50%', 'supply_flow_75%', 'supply_flow_max', 'supply_flow_head_5_mean', 'supply_flow_tail_5_mean', 'supply_flow_mad', 'supply_pressure_mean', 'supply_pressure_std', 'supply_pressure_min', 'supply_pressure_25%', 'supply_pressure_50%', 'supply_pressure_75%', 'supply_pressure_max', 'supply_pressure_head_5_mean', 'supply_pressure_tail_5_mean', 'supply_pressure_mad', 'return_temperature_mean', 'return_temperature_std', 'return_temperature_min', 'return_temperature_25%', 'return_temperature_50%', 'return_temperature_75%', 'return_temperature_max', 'return_temperature_head_5_mean', 'return_temperature_tail_5_mean', 'return_temperature_mad', 'return_conductivity_mean', 'return_conductivity_std', 'return_conductivity_min', 'return_conductivity_25%', 'return_conductivity_50%', 'return_conductivity_75%', 'return_conductivity_max', 'return_conductivity_head_5_mean', 'return_conductivity_tail_5_mean', 'return_conductivity_mad', 'return_turbidity_mean', 'return_turbidity_std', 'return_turbidity_min', 'return_turbidity_25%', 'return_turbidity_50%', 'return_turbidity_75%', 'return_turbidity_max', 'return_turbidity_head_5_mean', 'return_turbidity_tail_5_mean', 'return_turbidity_mad', 'return_flow_mean', 'return_flow_std', 'return_flow_min', 'return_flow_25%', 'return_flow_50%', 'return_flow_75%', 'return_flow_max', 'return_flow_head_5_mean', 'return_flow_tail_5_mean', 'return_flow_mad', 'tank_level_pre_rinse_mean', 'tank_level_pre_rinse_std', 'tank_level_pre_rinse_min', 'tank_level_pre_rinse_25%', 'tank_level_pre_rinse_50%', 'tank_level_pre_rinse_75%', 'tank_level_pre_rinse_max', 'tank_level_pre_rinse_head_5_mean', 'tank_level_pre_rinse_tail_5_mean', 'tank_level_pre_rinse_mad', 'tank_level_caustic_mean', 'tank_level_caustic_std', 'tank_level_caustic_min', 'tank_level_caustic_25%', 'tank_level_caustic_50%', 'tank_level_caustic_75%', 'tank_level_caustic_max', 'tank_level_caustic_head_5_mean', 'tank_level_caustic_tail_5_mean', 'tank_level_caustic_mad', 'tank_level_acid_mean', 'tank_level_acid_std', 'tank_level_acid_min', 'tank_level_acid_25%', 'tank_level_acid_50%', 'tank_level_acid_75%', 'tank_level_acid_max', 'tank_level_acid_head_5_mean', 'tank_level_acid_tail_5_mean', 'tank_level_acid_mad', 'tank_level_clean_water_mean', 'tank_level_clean_water_std', 'tank_level_clean_water_min', 'tank_level_clean_water_25%', 'tank_level_clean_water_50%', 'tank_level_clean_water_75%', 'tank_level_clean_water_max', 'tank_level_clean_water_head_5_mean', 'tank_level_clean_water_tail_5_mean', 'tank_level_clean_water_mad', 'tank_temperature_pre_rinse_mean', 'tank_temperature_pre_rinse_std', 'tank_temperature_pre_rinse_min', 'tank_temperature_pre_rinse_25%', 'tank_temperature_pre_rinse_50%', 'tank_temperature_pre_rinse_75%', 'tank_temperature_pre_rinse_max', 'tank_temperature_pre_rinse_head_5_mean', 'tank_temperature_pre_rinse_tail_5_mean', 'tank_temperature_pre_rinse_mad', 'tank_temperature_caustic_mean',
                        'tank_temperature_caustic_std', 'tank_temperature_caustic_min', 'tank_temperature_caustic_25%', 'tank_temperature_caustic_50%', 'tank_temperature_caustic_75%', 'tank_temperature_caustic_max', 'tank_temperature_caustic_head_5_mean', 'tank_temperature_caustic_tail_5_mean', 'tank_temperature_caustic_mad', 'tank_temperature_acid_mean', 'tank_temperature_acid_std', 'tank_temperature_acid_min', 'tank_temperature_acid_25%', 'tank_temperature_acid_50%', 'tank_temperature_acid_75%', 'tank_temperature_acid_max', 'tank_temperature_acid_head_5_mean', 'tank_temperature_acid_tail_5_mean', 'tank_temperature_acid_mad', 'tank_concentration_caustic_mean', 'tank_concentration_caustic_std', 'tank_concentration_caustic_min', 'tank_concentration_caustic_25%', 'tank_concentration_caustic_50%', 'tank_concentration_caustic_75%', 'tank_concentration_caustic_max', 'tank_concentration_caustic_head_5_mean', 'tank_concentration_caustic_tail_5_mean', 'tank_concentration_caustic_mad', 'tank_concentration_acid_mean', 'tank_concentration_acid_std', 'tank_concentration_acid_min', 'tank_concentration_acid_25%', 'tank_concentration_acid_50%', 'tank_concentration_acid_75%', 'tank_concentration_acid_max', 'tank_concentration_acid_head_5_mean', 'tank_concentration_acid_tail_5_mean', 'tank_concentration_acid_mad', 'total_turbidity_liter_mean', 'total_turbidity_liter_std', 'total_turbidity_liter_min', 'total_turbidity_liter_25%', 'total_turbidity_liter_50%', 'total_turbidity_liter_75%', 'total_turbidity_liter_max', 'total_turbidity_liter_head_5_mean', 'total_turbidity_liter_tail_5_mean', 'total_turbidity_liter_mad', 'supply_pump_mean', 'supply_pump_head_5_mean', 'supply_pump_tail_5_mean', 'supply_pre_rinse_mean', 'supply_pre_rinse_head_5_mean', 'supply_pre_rinse_tail_5_mean', 'supply_caustic_mean', 'supply_caustic_head_5_mean', 'supply_caustic_tail_5_mean', 'return_caustic_mean', 'return_caustic_head_5_mean', 'return_caustic_tail_5_mean', 'supply_acid_mean', 'supply_acid_head_5_mean', 'supply_acid_tail_5_mean', 'return_acid_mean', 'return_acid_head_5_mean', 'return_acid_tail_5_mean', 'supply_clean_water_mean', 'supply_clean_water_head_5_mean', 'supply_clean_water_tail_5_mean', 'return_recovery_water_mean', 'return_recovery_water_head_5_mean', 'return_recovery_water_tail_5_mean', 'return_drain_mean', 'return_drain_head_5_mean', 'return_drain_tail_5_mean', 'object_low_level_mean', 'object_low_level_head_5_mean', 'object_low_level_tail_5_mean', 'tank_lsh_caustic_mean', 'tank_lsh_caustic_head_5_mean', 'tank_lsh_caustic_tail_5_mean', 'tank_lsh_acid_mean', 'tank_lsh_acid_head_5_mean', 'tank_lsh_acid_tail_5_mean', 'tank_lsh_clean_water_mean', 'tank_lsh_clean_water_head_5_mean', 'tank_lsh_clean_water_tail_5_mean', 'tank_lsh_pre_rinse_mean', 'tank_lsh_pre_rinse_head_5_mean', 'tank_lsh_pre_rinse_tail_5_mean', 'total_turbidity_liter_sum', 'pre_rinse', 'caustic', 'intermediate_rinse', 'acid', 'pipeline_L1', 'pipeline_L10', 'pipeline_L11', 'pipeline_L12', 'pipeline_L2', 'pipeline_L3', 'pipeline_L4', 'pipeline_L6', 'pipeline_L7', 'pipeline_L8', 'pipeline_L9']
for column in model_phase_features:
    if(column not in inputData_phases.columns):
        inputData_phases[column] = 0

print('Loading models....')
xgb_model_phases = {}
xgb_model_phases_lower = {}
for i in range(1, 16):
    xgb_model_phases[str(i)] = load(
        open(join(models_directory, 'xgb-model-phase-'+str(i)+'.dat'), 'rb'))
    xgb_model_phases_lower[str(i)+'_lower'] = load(
        open(join(models_directory, 'xgb-model-phase-'+str(i)+'_lower.dat'), 'rb'))

inputData_phases = inputData_phases.groupby('phases_code')
train_pred_dict = dict()
for inputData_phase in inputData_phases:
    phases_code = str(inputData_phase[0])
    target_squared_inverse_sum = target_squared_inverse_sums[phases_code]
    input_values = inputData_phase[1]
    input_values = input_values[model_phase_features]
    xgb_final_rinse_total_turbidity_liter_pred = xgb_model_phases[phases_code].predict(
        input_values)
    xgb_final_rinse_total_turbidity_liter_pred = [
        (1/((value**2) * target_squared_inverse_sum)) for value in xgb_final_rinse_total_turbidity_liter_pred]
    # check for lower values
    indices = []
    index = 0
    for value in xgb_final_rinse_total_turbidity_liter_pred:
        if(value < threshold):
            indices.append(index)
        index += 1

    if(len(indices) > 0):
        input_values = input_values.iloc[indices]
        xgb_final_rinse_total_turbidity_liter_pred_lower = xgb_model_phases_lower[phases_code+"_lower"].predict(
            input_values)
        xgb_final_rinse_total_turbidity_liter_pred_lower = [
            value**2 for value in xgb_final_rinse_total_turbidity_liter_pred_lower]
        temp_index = 0
        for index in indices:
            xgb_final_rinse_total_turbidity_liter_pred[
                index] = xgb_final_rinse_total_turbidity_liter_pred_lower[temp_index]
            temp_index += 1

    print('grouping predictions for phases code:', phases_code)
    for process_id, xgb_final_rinse_total_turbidity_liter_value in zip(inputData_phase[1]['process_id'].values, xgb_final_rinse_total_turbidity_liter_pred):
        if(process_id not in train_pred_dict):
            train_pred_dict[process_id] = {}

        train_pred_dict[process_id][phases_code] = scale_value(
            xgb_final_rinse_total_turbidity_liter_value)


data_table = [['process_id', 'final_rinse_total_turbidity_liter']]

for process_id in sorted(train_pred_dict.keys()):
    data_table.append([process_id, np.median(list(train_pred_dict[process_id].values()))])

# write data to file
import csv
print('Writing to csv.........')
with open(join(output_directory, path.split(reshaped_test_data_path)[1].split(
        '.')[0]+'-predictions.csv'), 'w+') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',', lineterminator='\n')
    csvWriter.writerows(data_table)

print('Finished prediction')
print('--- %s seconds ---' % ((time() - start_time)))
