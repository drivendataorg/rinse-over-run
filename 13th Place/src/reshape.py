# %% import libraries and set input parameters
from time import time
start_time = time()

from os import listdir, path
from os.path import isfile, join, dirname, realpath
from pandas import DataFrame
import xgboost as xgb
from pickle import load
from sys import argv
from multiprocessing import cpu_count, Pool
import pandas as pd
import numpy as np

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


raw_data_path = arguments['--raw_data_path']
input_labels_path = None
if('--input_labels_path' in arguments):
    input_labels_path = arguments['--input_labels_path']

output_directory = arguments['--output_directory']

print('raw_data_path, input_labels_path, output_directory: ',
      raw_data_path, input_labels_path, output_directory)

final_rinse_total_turbidity_liter = 'final_rinse_total_turbidity_liter'

def skip_rows(index):
    # return index % 1000 > 0# for faster testing of code
    return False  # load all


# each phase flag can be set to 0 or 1
# acid - intermediate_rinse - caustic - pre_rinse
# example: 1100 means that it represents acid & intermediate_rinse
phases_indices = {'pre_rinse': 0, 'caustic': 1,
                  'intermediate_rinse': 2, 'acid': 3}


def reshape(params):
    input_values_dfs, columns_numeric, columns_flags = params
    final_data = []
    for input_values_df in input_values_dfs:
        input_values_df = input_values_df.reset_index(drop=True)

        process_id = input_values_df.loc[0, 'process_id']
        pipeline = input_values_df.loc[0, 'pipeline']
        print(process_id, pipeline)

        phases_dfs = [None, None, None, None]
        # group by phase
        process_phases = input_values_df.groupby('phase')
        phases_names = []
        for process_phase_info in process_phases:
            phase_df = process_phase_info[1].reset_index(drop=True)
            phases_dfs[phases_indices[process_phase_info[0]]] = phase_df
            phases_names.append(process_phase_info[0])

        for phases_code in range(1, 16):
            include_pre_rinse = (phases_code & 1 == 1)
            include_caustic = (phases_code & 2 == 2)
            include_intermediate_rinse = (phases_code & 4 == 4)
            include_acid = (phases_code & 8 == 8)
            phases_union_dfs = []
            if(include_pre_rinse):
                if(phases_dfs[0] is not None):
                    phases_union_dfs.append(phases_dfs[0])
                else:
                    continue

            if(include_caustic):
                if(phases_dfs[1] is not None):
                    phases_union_dfs.append(phases_dfs[1])
                else:
                    continue

            if(include_intermediate_rinse):
                if(phases_dfs[2] is not None):
                    phases_union_dfs.append(phases_dfs[2])
                else:
                    continue

            if(include_acid):
                if(phases_dfs[3] is not None):
                    phases_union_dfs.append(phases_dfs[3])
                else:
                    continue

            # union dfs
            phases_union_df = pd.DataFrame(
                [], columns=phases_union_dfs[0].columns)
            for df in phases_union_dfs:
                phases_union_df = phases_union_df.append(df)

            total_count = phases_union_df.shape[0]
            phases_union_df = phases_union_df.reset_index(
                drop=True)
            # process_id, phase, pipeline
            phases_union_record = [
                process_id, pipeline, phases_code, total_count, '.'.join(phases_names)]
            for col_name in columns_numeric:
                col_values = phases_union_df[col_name]
                stats = col_values.describe()
                phases_union_record.append(stats['mean'])
                phases_union_record.append(stats['std'])
                phases_union_record.append(stats['min'])
                phases_union_record.append(stats['25%'])
                phases_union_record.append(stats['50%'])
                phases_union_record.append(stats['75%'])
                phases_union_record.append(stats['max'])
                phases_union_record.append(col_values.head(5).mean())
                phases_union_record.append(col_values.tail(5).mean())
                phases_union_record.append(col_values.mad())

            for col_name in columns_flags:
                col_values = phases_union_df[col_name]
                phases_union_record.append(col_values.mean())
                phases_union_record.append(col_values.head(5).mean())
                phases_union_record.append(col_values.tail(5).mean())

            phases_union_record.append(
                phases_union_df['total_turbidity_liter'].sum())

            if(input_labels_path is not None):
                phases_union_record.append(
                    phases_union_df.loc[0, final_rinse_total_turbidity_liter])

            final_data.append(phases_union_record)

    return final_data


if __name__ == '__main__':
    # for training our model
    print("Loading data...")
    input_values_all = pd.read_csv(raw_data_path, skiprows=skip_rows)
    print(input_values_all.shape)

    # drop columns that are not valued
    input_values_all = input_values_all.drop(
        ['row_id', 'object_id', 'timestamp', 'target_time_period'], axis=1)

    # get data  without final_rinse
    input_values_all = input_values_all[input_values_all.phase != 'final_rinse']
    print(input_values_all.shape)

    # change bool columns to integer columns for getting statistics
    for col_name in input_values_all.columns:
        if(input_values_all[col_name].dtype == np.bool):
            input_values_all[col_name] = input_values_all[col_name].astype(int)

    # set target variable
    if(input_labels_path is not None):
        input_labels = pd.read_csv(input_labels_path)
        # convert train labels to dictionary
        input_labels_dict = input_labels.set_index(
            'process_id')[final_rinse_total_turbidity_liter].to_dict()
        # append final_rinse_total_turbidity_liter to data
        input_values_all[final_rinse_total_turbidity_liter] = input_values_all['process_id'].map(
            lambda value: input_labels_dict[value])

    # create field similar to target
    input_values_all['total_turbidity_liter'] = np.maximum(
        input_values_all.return_flow, 0) * input_values_all.return_turbidity

    # run threads to reshape data
    threads_count = cpu_count()
    po = Pool(threads_count)
    all_input_values_parts = []
    input_values_processes_groups = input_values_all.groupby('process_id')
    input_values_processes_dfs = []
    for g in input_values_processes_groups:
        input_values_processes_dfs.append(g[1])

    print('input_values_processes_dfs', len(input_values_processes_dfs))
    step = int(len(input_values_processes_dfs)/threads_count)
    if(len(input_values_processes_dfs) % threads_count > 0):  # step contains decimal point
        step += 1
    for i in range(threads_count):
        if(i < threads_count-1):
            all_input_values_parts.append(
                input_values_processes_dfs[i*step: (i+1)*step])
        else:
            all_input_values_parts.append(
                input_values_processes_dfs[i*step: len(input_values_processes_dfs)])

    columns_numeric = ['supply_flow', 'supply_pressure', 'return_temperature', 'return_conductivity', 'return_turbidity', 'return_flow', 'tank_level_pre_rinse', 'tank_level_caustic',
                       'tank_level_acid', 'tank_level_clean_water', 'tank_temperature_pre_rinse', 'tank_temperature_caustic', 'tank_temperature_acid', 'tank_concentration_caustic', 'tank_concentration_acid', 'total_turbidity_liter']

    columns_flags = ['supply_pump', 'supply_pre_rinse', 'supply_caustic', 'return_caustic', 'supply_acid',
                     'return_acid', 'supply_clean_water', 'return_recovery_water', 'return_drain', 'object_low_level', 'tank_lsh_caustic', 'tank_lsh_acid', 'tank_lsh_clean_water', 'tank_lsh_pre_rinse']

    final_data_all = po.map_async(reshape,
                                  ((item, columns_numeric, columns_flags)
                                   for item in all_input_values_parts)
                                  # this will zip in one iterable object
                                  ).get()  # get will start the processes and execute them
    po.terminate()  # kill the spawned processes

    final_columns = ['process_id', 'pipeline',
                     'phases_code', 'count', 'phases_names']
    for col_name in columns_numeric:
        final_columns.append(col_name+'_mean')
        final_columns.append(col_name+'_std')
        final_columns.append(col_name+'_min')
        final_columns.append(col_name+'_25%')
        final_columns.append(col_name+'_50%')
        final_columns.append(col_name+'_75%')
        final_columns.append(col_name+'_max')
        final_columns.append(col_name+'_head_5_mean')
        final_columns.append(col_name+'_tail_5_mean')
        final_columns.append(col_name+'_mad')

    for col_name in columns_flags:
        final_columns.append(col_name+'_mean')
        final_columns.append(col_name+'_head_5_mean')
        final_columns.append(col_name+'_tail_5_mean')

    final_columns.append('total_turbidity_liter_sum')
    if(input_labels_path is not None):
        final_columns.append(final_rinse_total_turbidity_liter)

    final_data_all_union = []

    for final_data in final_data_all:
        for record in final_data:
            final_data_all_union.append(record)

    final_final_df = pd.DataFrame(
        data=final_data_all_union, columns=final_columns)
    final_final_df.to_csv(join(output_directory, path.split(raw_data_path)[1].split(
        '.')[0]+'-reshaped-phases-combined.csv'), index=False)
    seconds = ((time() - start_time))
    print('seconds', seconds, 'minutes', round(seconds/60, 3))
    print('Finished')
