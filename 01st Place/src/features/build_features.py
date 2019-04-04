# The essentials
import pandas as pd
import numpy as np

# CLI & Logging
import click
import logging

# tsfresh Feature Extraction
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute


def encode_categorical(df):
    """Process the different categorical data in df (almost identical
    to the baseline from DrivenData).

    Parameters:
    -----------
    - df: pd.DataFrame
        the raw (timeseries) data containing the categorical data

    Returns:
    --------
    - meta: pd.DataFrame
        a DataFrame with each record a process, containing the features
        based on the categorical meta-data
    """
    # select process_id, pipeline and object_id
    meta = df[['process_id', 'pipeline', 'object_id']]
    meta = meta.drop_duplicates().set_index('process_id')

    # bin the process_ids per 5
    meta['object_id'] = meta['object_id'] // 5

    # convert categorical pipeline data to dummy variables
    meta = pd.get_dummies(meta, columns=['pipeline', 'object_id'])

    # pipeline L12 not in test data (so useless feature)
    if 'pipeline_L12' in meta:
        meta = meta.drop('pipeline_L12', axis=1)

    return meta


def count_zeros(x):
    return np.sum(x == 0)


def encode_real_timeseries(df):
    """Calculate different aggregates/descriptive statistics, using pandas,
    of the real-valued raw timeseries.

    Parameters:
    -----------
    - df: pd.DataFrame
        the raw (timeseries) data containing the categorical features

    Returns:
    --------
    - ts_features: pd.DataFrame
        a DataFrame with each record a process, containing the features
        based on the real-valued timeseries
    """
    real_cols = [
        'supply_flow', 'supply_pressure', 'return_temperature',
        'return_conductivity', 'return_turbidity', 'return_flow',
        'tank_level_pre_rinse', 'tank_level_caustic', 'tank_level_acid',
        'tank_level_clean_water', 'tank_temperature_pre_rinse',
        'tank_temperature_caustic', 'tank_temperature_acid',
        'tank_concentration_caustic', 'tank_concentration_acid',
        'target_value', 'flow_diff'
    ]

    flow_cols = [
        'supply_flow',
        'return_flow',
        'target_value'
    ]

    ts_df = df[['process_id'] + real_cols].set_index('process_id')
    
    # Group by process_id and extract different statistics
    ts_features = ts_df.groupby('process_id').agg(['min', 'max', 'mean', 'std', 
                                                   'count', 'median', 'sum', 
                                                   lambda x: x.tail(5).mean(),
                                                   count_zeros])

    # Rename the columns so we can later easily identify that the features
    # have been extracted by this function
    cols = []
    for col in ts_features.columns:
        cols.append('real_{}'.format(col))
    ts_features.columns = cols
    
    # Remove outliers in the different "flow" columns and calculate basic
    # aggregates
    flow_df = df[['process_id', 'object_id'] + flow_cols]
    flow_df = flow_df.reset_index(drop=True)
    for machine in set(flow_df['object_id']):
        mach_data = flow_df[flow_df['object_id'] == machine]
        for col in flow_cols:
            perc = np.percentile(mach_data[col], 99)
            flow_df.loc[mach_data.index, :][col] = mach_data[col].clip(0, perc)
    flow_df = flow_df.set_index('process_id')
    flow_df = flow_df.drop('object_id', axis=1)
    flow_features = flow_df.groupby('process_id').agg(['max', 'mean', 'sum'])
    
    # Again rename these columns
    cols = []
    for col in flow_features.columns:
        cols.append('flow_{}'.format(col))
    flow_features.columns = cols
    
    # Join both extracted features together
    ts_features = ts_features.merge(flow_features, left_index=True, 
                                    right_index=True)
    
    return ts_features


def encode_binary_timeseries(df):
    """Calculate different aggregates/descriptive statistics, using pandas,
    of the binary-valued raw timeseries.

    Parameters:
    -----------
    - df: pd.DataFrame
        the raw (timeseries) data containing the categorical features

    Returns:
    --------
    - ts_features: pd.DataFrame
        a DataFrame with each record a process, containing the features
        based on the binary-valued timeseries
    """
    bin_cols = [
        'supply_pump', 'supply_pre_rinse', 'supply_caustic', 'return_caustic',
        'supply_acid', 'return_acid', 'supply_clean_water',
        'return_recovery_water', 'return_drain', 'object_low_level',
        'tank_lsh_caustic', 'tank_lsh_acid', 'tank_lsh_clean_water',
        'tank_lsh_pre_rinse'
    ]

    ts_df = df[['process_id'] + bin_cols].set_index('process_id')
            
    # create features: mean, standard deviation, mean of final values,
    # sum and number of zeros
    ts_features = ts_df.groupby('process_id').agg(['mean', 'std', 'sum',
                                                   lambda x: x.tail(5).mean(),
                                                   count_zeros])
    
    cols = []
    for col in ts_features.columns:
        cols.append('bin_{}'.format(col))
    ts_features.columns = cols
    
    return ts_features


def get_tsfresh_features(df):
    """Calculate different aggregates/descriptive statistics, using tsfresh,
    of the some of the more informative raw timeseries.

    Parameters:
    -----------
    - df: pd.DataFrame
        the raw (timeseries) data containing the categorical features

    Returns:
    --------
    - ts_features: pd.DataFrame
        a DataFrame with each record a process, containing the features
        based on the binary-valued timeseries
    """

    # We only keep the feature extraction functions that are not too
    # computationally expensive & that do not return too many values
    extraction_settings = EfficientFCParameters()
    filtered_funcs = [
        'abs_energy', 'mean_abs_change', 'mean_change', 'skewness', 
        'kurtosis', 'absolute_sum_of_changes', 'longest_strike_below_mean', 
        'longest_strike_above_mean', 'count_above_mean', 'count_below_mean', 
        'last_location_of_maximum', 'first_location_of_maximum', 
        'last_location_of_minimum', 'first_location_of_minimum', 
        'percentage_of_reoccurring_datapoints_to_all_datapoints', 
        'percentage_of_reoccurring_values_to_all_values', 
        'sum_of_reoccurring_values', 'sum_of_reoccurring_data_points', 
        'ratio_value_number_to_time_series_length', 'cid_ce', 
        'symmetry_looking', 'large_standard_deviation', 'quantile', 
        'autocorrelation', 'number_peaks', 'binned_entropy', 
        'index_mass_quantile', 'linear_trend',  'number_crossing_m', 
        'augmented_dickey_fuller', 'number_cwt_peaks', 'agg_autocorrelation', 
        'spkt_welch_density', 'friedrich_coefficients', 
        'max_langevin_fixed_point', 'c3', 'ar_coefficient', 
        'mean_second_derivative_central', 'ratio_beyond_r_sigma', 
        'energy_ratio_by_chunks', 'partial_autocorrelation', 'fft_aggregated', 
        'time_reversal_asymmetry_statistic', 'range_count'
    ]
    filtered_settings = {}
    for func in filtered_funcs:
      filtered_settings[func] = extraction_settings[func]

    # Extract the features
    ts_features = extract_features(df[['process_id', 'timestamp', 
                                       'return_turbidity', 'return_flow', 
                                       'supply_flow', 'target_value', 
                                       'flow_diff']], 
                                   column_id='process_id', 
                                   column_sort="timestamp", 
                                   column_kind=None, column_value=None,
                                   impute_function=impute, 
                                   default_fc_parameters=filtered_settings,
                                   show_warnings=False, 
                                   disable_progressbar=True)
  
    return ts_features


def create_feature_matrix(df, processes, phases):
    """Calculate all features for certain processes and certain phases of
    these processes, located in a provided dataframe.

    Parameters:
    -----------
    - df: pd.DataFrame
        the raw (timeseries) data
    - processes: array-like
        the processes for which to calculate the features
    - phases: array-like
        the phases to use for each of the processes

    Returns:
    --------
    - feature_matrix: pd.DataFrame
        a DataFrame with each record a process, containing all features
    """
    # Filter out the right data
    phase_data = df[(df['process_id'].isin(processes)) &
                    ((df['phase'].isin(phases)))].copy()

    # Calculate two new timeseries, one the product of return turbidity and
    # return flow, which corresponds to the target (w/o target_time_period);
    # and one based on the different in supply and return flow.
    phase_data.loc[:, 'return_flow'] = phase_data['return_flow'].apply(lambda x: max(x, 0))
    phase_data.loc[:, 'supply_flow'] = phase_data['supply_flow'].apply(lambda x: max(x, 0))
    phase_data.loc[:, 'target_value'] = phase_data['return_flow'] * phase_data['return_turbidity']
    phase_data.loc[:, 'flow_diff'] = phase_data['supply_flow'] - phase_data['return_flow']
    
    # Call the different feature extraction functions
    logging.info('Extracting categorical features...')
    metadata = encode_categorical(phase_data)
    logging.info('Extracting real-valued ts features with pandas...')
    time_series = encode_real_timeseries(phase_data)
    logging.info('Extracting binary-valued ts features with pandas...')
    binary_features = encode_binary_timeseries(phase_data)
    logging.info('Extracting real-valued ts features with tsfresh...')
    tsfresh_features = get_tsfresh_features(phase_data)
    
    # Calculate some statistics based on data from solely the final phase
    # of the present data
    if len(phases) > 1:
      last_phase_data = phase_data[phase_data['phase'] == phases[-1]]
      time_series_last_phase = encode_real_timeseries(last_phase_data)
      new_cols = []
      for col in time_series_last_phase.columns:
        new_cols.append('last_{}'.format(col))
      time_series_last_phase.columns = new_cols
      binary_features_last_phase = encode_binary_timeseries(last_phase_data)
      new_cols = []
      for col in binary_features_last_phase.columns:
        new_cols.append('last_{}'.format(col))
      binary_features_last_phase.columns = new_cols
    
    # Join all the different feature dataframes together
    feature_matrix = metadata
    feature_matrix = feature_matrix.merge(time_series, left_index=True, 
                                          right_index=True)
    feature_matrix = feature_matrix.merge(binary_features, left_index=True, 
                                          right_index=True)
    feature_matrix = feature_matrix.merge(tsfresh_features, left_index=True, 
                                          right_index=True)
    if len(phases) > 1:
        feature_matrix = feature_matrix.merge(time_series_last_phase, 
                                            left_index=True, right_index=True)
        feature_matrix = feature_matrix.merge(binary_features_last_phase, 
                                            left_index=True, right_index=True)
    
    return feature_matrix

def get_processes(data, phases, train=True):
    """Get the processes for which certain phases are present. For train
    processes, the provided phases must be a subset of the present phases;
    for the test set, the provided and present phases must exactly match.

    Parameters
    ----------
    - data: pd.DataFrame
        the raw data 
    - phases: array-like
        the phases that (at least) need to be present in the data
    -train: bool
        whether or not we must extract processes from the training or 
        testing data
    
    Returns
    -------
    - filtered_processes: array-like

    """
    filtered_processes = []
    phases = set(phases)
    processes = set(data['process_id'])
    for process in processes:
        process_phases = set(data[data['process_id'] == process]['phase'])
        if train:
            if phases.issubset(process_phases):
                filtered_processes.append(process)
        else:
            if len(phases) == len(process_phases) == len(phases.intersection(process_phases)):
                filtered_processes.append(process)
    return filtered_processes

def load_data(train_path, test_path, label_path, recipe_path):
    """Load the different input dataframes.

    Parameters:
    -----------
    - train_path:
        the location of the raw data of the train processes
    - test_path:
        the location of the raw data of the test processes
    - label_path:
        the location of the labels for the train processes
    - recipe_path:
        the location of recipe metadata for all processes

    Returns
    -------
    - train_df: pd.DataFrame
        the raw data of the train processes
    - test_df: pd.DataFrame
        the raw data of the test processes
    - label_df: pd.DataFrame
        the target values for the train processes
    - recipe_df: pd.DataFrame
        the recipe metadata of both train and test processes
    """
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=['timestamp'])
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=['timestamp'])
    label_df = pd.read_csv(label_path, index_col='process_id')

    recipe_df = pd.read_csv(recipe_path, index_col='process_id')
    recipe_df = recipe_df.drop('final_rinse', axis=1)
    recipe_df['pre_rinse_num'] = recipe_df['pre_rinse'] * 1
    recipe_df['caustic_num'] = recipe_df['caustic'] * 2
    recipe_df['intermediate_rinse_num'] = recipe_df['intermediate_rinse'] * 4
    recipe_df['acid_num'] = recipe_df['acid'] * 8
    recipe_df['recipe'] = recipe_df['pre_rinse_num'] + recipe_df['caustic_num'] + recipe_df['intermediate_rinse_num'] + recipe_df['acid_num']

    return train_df, test_df, label_df, recipe_df

def get_corr_features(X):
    """Get all coordinates in the X-matrix with correlation value equals 1
    (columns with equal values), excluding elements on the diagonal.

    Parameters:
    -----------
    - train_df: pd.DataFrame
        the feature matrix where correlated features need to be removed

    Returns
    -------
    - correlated_feature_pairs: list of tuples
        coordinates (row, col) where correlated features can be found
    """
    row_idx, col_idx = np.where(X.corr() == 1)
    self_corr = set([(i, i) for i in range(X.shape[1])])
    correlated_feature_pairs = set(list(zip(row_idx, col_idx))) - self_corr
    return correlated_feature_pairs


def get_uncorr_features(data):
    """Remove clusters of these correlated features, until only one feature 
    per cluster remains.

    Parameters:
    -----------
    - data: pd.DataFrame
        the feature matrix where correlated features need to be removed

    Returns
    -------
    - data_uncorr_cols: list of string
        the column names that are completely uncorrelated to eachother
    """
    X_train_corr = data.copy()
    correlated_features = get_corr_features(X_train_corr)

    corr_cols = set()
    for row_idx, col_idx in correlated_features:
        corr_cols.add(row_idx)
        corr_cols.add(col_idx)

    uncorr_cols = list(set(X_train_corr.columns) - set(X_train_corr.columns[list(corr_cols)]))
   
    col_mask = [False]*X_train_corr.shape[1]
    for col in corr_cols:
        col_mask[col] = True
    X_train_corr = X_train_corr.loc[:, col_mask]
  
    correlated_features = get_corr_features(X_train_corr)

    while correlated_features:
        corr_row, corr_col = correlated_features.pop()
        col_mask = [True]*X_train_corr.shape[1]
        col_mask[corr_row] = False
        X_train_corr = X_train_corr.loc[:, col_mask]
        correlated_features = get_corr_features(X_train_corr)

    data_uncorr_cols = list(set(list(X_train_corr.columns) + uncorr_cols))

    return data_uncorr_cols

def remove_features(data):
    """Remove all correlated features and columns with only a single value.

    Parameters:
    -----------
    - data: pd.DataFrame
        the feature matrix where correlated features need to be removed

    Returns
    -------
    - useless_cols: list of string
        list of column names that have no predictive value
    """
    single_cols = list(data.columns[data.nunique() == 1])

    uncorr_cols = get_uncorr_features(data)
    corr_cols = list(set(data.columns) - set(uncorr_cols))

    useless_cols = list(set(single_cols + corr_cols))

    logging.info('Removing {} features'.format(len(useless_cols)))

    return useless_cols

@click.command()
@click.argument('train_path', type=click.Path(exists=True))
@click.argument('test_path', type=click.Path(exists=True))
@click.argument('label_path', type=click.Path(exists=True))
@click.argument('recipe_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
def main(train_path, test_path, label_path, recipe_path, output_path):
    """Takes the raw data located at different paths and produces different 
    feature files, one for each unique (recipe, present phases)-combination.
    The feature files will be located at data/features.

    Parameters:
    -----------
    - train_path:
        the location of the raw data of the train processes
    - test_path:
        the location of the raw data of the test processes
    - label_path:
        the location of the labels for the train processes
    - recipe_path:
        the location of recipe metadata for all processes
    - output_path:
        the location where to store the extracted features
    """
    process_comb_to_phases = {
        15: ['pre_rinse', 'caustic', 'intermediate_rinse', 'acid'],
        3:  ['pre_rinse', 'caustic'],
        7:  ['pre_rinse', 'caustic', 'intermediate_rinse'],
        1:  ['pre_rinse'],
        8:  ['acid'],
        2:  ['caustic'],
        6:  ['caustic', 'intermediate_rinse'],
        14: ['caustic', 'intermediate_rinse', 'acid'],
    }

    combinations_per_recipe = {
        3: [1, 2, 3], 
        9: [8],
        15: [1, 2, 3, 6, 7, 14, 15]
    }

    # Load the data
    train_df, test_df, label_df, recipe_df = load_data(train_path, test_path, 
                                                       label_path, 
                                                       recipe_path)
    all_data = pd.concat([train_df, test_df], axis=0)

    # Two for-loops to get the processes for each unique 
    # (recipe, present phases)-combination
    for recipe in [3, 9, 15]:
        recipe_train_data = train_df[train_df['process_id'].isin(recipe_df[recipe_df['recipe'] == recipe].index)]
        recipe_test_data = test_df[test_df['process_id'].isin(recipe_df[recipe_df['recipe'] == recipe].index)]
        for process_combination in combinations_per_recipe[recipe]:
            logging.info('Extracting features for ({}, {})-combination...'.format(recipe, process_combination))
            train_processes = get_processes(recipe_train_data, process_comb_to_phases[process_combination])
            test_processes = get_processes(recipe_test_data, process_comb_to_phases[process_combination], train=False)
            logging.info('#Training processes = {} || #Testing processes = {}'.format(len(train_processes), len(test_processes)))

            # For (9, 8) we perform some data augmentation by also using
            # processes of recipe 15 in the training set.
            if (recipe, process_combination) in [(9, 8)]:
                recipe_15_train_data = train_df[train_df['process_id'].isin(recipe_df[recipe_df['recipe'] == 15].index)]
                extra_processes = get_processes(recipe_15_train_data, process_comb_to_phases[process_combination])
                train_processes += extra_processes

            # We extract features for all data at once, since it is 
            # completely unsupervised, and each row is processed independently
            all_processes = train_processes + test_processes
            phase_features = create_feature_matrix(all_data, all_processes, process_comb_to_phases[process_combination])

            # Drop columns without any predictive value
            to_drop = remove_features(phase_features)
            phase_features = phase_features.drop(to_drop, axis=1)

            # Extract the train and test features
            train_phase_features = phase_features.loc[train_processes]
            train_phase_features['target'] = label_df.loc[train_phase_features.index]['final_rinse_total_turbidity_liter']
            test_phase_features = phase_features.loc[test_processes]

            logging.info('#Training feature matrix: {} || #Testing feature matrix: {}'.format(train_phase_features.shape, test_phase_features.shape))

            # Write them to the provided output_path
            train_phase_features.to_csv('{}/train_features_{}_{}.csv'.format(output_path, recipe, process_combination))
            test_phase_features.to_csv('{}/test_features_{}_{}.csv'.format(output_path, recipe, process_combination))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
