import pandas as pd
import numpy as np
import logging as logger

pd.options.mode.chained_assignment = None


def ingest_data(path):
    """Reads in data from source files in pandas dataframes.

    The following files are assumed to be present in the directory specified by the 'path' argument:
    train_values.pkl: train_values.csv converted to .pkl format to reduce load times.
    test_values.pkl: test_values.csv converted to .pkl format to reduce load times.
    train_labels.csv
    recipe_metadata.csv

    Also calculates the start times for each process, which are necessary for walk-forward/rolling origin validation.

    Args:
        path (str): directory where source files are located.

    Returns:
        The following dataframes in the listed order:
        raw_data (dataframe): training data for model building
        test_data (dataframe): test data for predictions
        labels (dataframe): response values for each process in the training data
        metadata (dataframe): recipe types for each process
        start_times (dataframe): start times for each process in the training and test datasets

    """

    # Read in data from source files
    logger.info('Raw data not found; reading in raw data from source files...')
    raw_data = pd.read_pickle(path + 'train_values.pkl')
    test_data = pd.read_pickle(path + 'test_values.pkl')
    labels = pd.read_csv(path + 'train_labels.csv')
    metadata = pd.read_csv(path + 'recipe_metadata.csv')
    logger.info('Successfully read in source data.')

    # Determine when each process started for both train and test data
    # Necessary to properly do walk forward validation and for feature engineering
    logger.info('Calculating process start times...')
    train_start_times = calculate_start_times(raw_data)
    test_start_times = calculate_start_times(test_data)
    start_times = pd.concat([train_start_times, test_start_times]).sort_values(by='start_time')
    logger.info('Process start times successfully calculated.')

    return raw_data, test_data, labels, metadata, start_times


def preprocess_data(df, test_data, start_times, return_phase_defs=None, supply_phase_defs=None):
    """Pre-processes raw train and test data before feature engineering and modeling.

    Performs the following pre-processing steps:
    1. Shortens lengthy phase names for conciseness.
    2. Removes objects that aren't in test set.
    3. Calculates 'return-phases' and 'supply-phases' for each timestamp (see modeling report for details).
    4. Calculates various process-timestamp-level (i.e. row-level in the original data) features.

    The valid values for 'return-phases' and 'supply-phases' are determined by their frequencies in the training data.
    Therefore, the training data must be pre-processed before the test data.

    In addition, the valid return-phases and supply-phases must be passed as lists using the return_phase_defs and
        supply_phase_defs params when preprocessing test data.

    Args:
        df (dataframe): dataframe of source data (process-timestamp level) to be pre-processed.
            Can be train or test data.
        test_data (dataframe): dataframe of test data.
        start_times (dataframe): start times for each process.
        return_phase_defs (list of str): list of valid return-phases. Only necessary for pre-processing test data.
            If None (default value), it is assumed that training data is being pre-processed, and a list of valid
            return-phases is returned.
        supply_phase_defs (list of str): list of valid supply-phases. Only necessary for pre-processing test data.

    Returns:
        df (dataframe): dataframe of pre-processed data.
        return_phases (list of str): a list of valid return-phases. Only returned if return_phase_defs arg is None.
            This list should be supplied as the value for the return_phase_defs arg when pre-processing test data.
        supply_phases (list of str): a list of valid supply-phases. Only returned if return_phase_defs arg is None.
            This list should be supplied as the value for the supply_phase_defs arg when pre-processing test data.
    """

    if return_phase_defs is None:
        data_type = 'training'
    else:
        data_type = 'test'

    logger.info('Pre-processing raw ' + data_type + ' data...')

    # Convert 'intermediate rinse' to 'int_rinse' for succinctness
    df.phase[df.phase == 'intermediate_rinse'] = 'int_rinse'

    # Remove processes with objects that aren't in test set
    df = df[df.object_id.isin(test_data.object_id)]

    df.timestamp = df.timestamp.astype('datetime64[s]')
    df = df.merge(start_times, on='process_id')

    logger.info('Calculating process-timestamp-level features...')
    # Return phase definition
    # Use shortened versions of phase names to avoid issues when simulating mid-process predictions
    # Simulation drops columns using regex that checks for full phase name
    df['return_phase'] = df.phase + '_' + np.where(df.return_drain == True, 'dr',
                                          np.where(df.return_caustic == True, 'cs',
                                          np.where(df.return_acid == True, 'ac',
                                          np.where(df.return_recovery_water == True, 'rw', 'none'))))

    # Bucket infrequent return phases as 'other'
    # Definition of 'infrequent' is determined by training set frequencies, which must be pre-computed and passed
    # as a parameter when pre-processing test set data
    if return_phase_defs is None:
        return_phases = list(
            df.return_phase.value_counts()[df.return_phase.value_counts() > 300000].reset_index()['index'])
    else:
        return_phases = return_phase_defs
    df['return_phase'] = np.where(df.return_phase.isin(return_phases), df.return_phase, 'other')

    # Supply phase definition
    df['supply_phase'] = df.phase + '_' + np.where(df.supply_pre_rinse == True, 'pr',
                                          np.where(df.supply_caustic == True, 'cs',
                                          np.where(df.supply_acid == True, 'ac',
                                          np.where(df.supply_clean_water == True, 'cw', 'none'))))

    # Bucket infrequent supply phases as 'other'
    # Same process as return phases
    if supply_phase_defs is None:
        supply_phases = list(
            df.supply_phase.value_counts()[df.supply_phase.value_counts() > 100000].reset_index()['index'])
    else:
        supply_phases = supply_phase_defs
    df['supply_phase'] = np.where(df.supply_phase.isin(supply_phases), df.supply_phase, 'other')

    # Other process-timestamp-level features
    df['return_flow'] = np.maximum(0, df.return_flow)
    df['supply_flow'] = np.maximum(0, df.supply_flow)
    df['return_residue'] = df.return_flow * df.return_turbidity
    df['phase_elapse_end'] = (
            df.groupby(['process_id', 'phase']).timestamp.transform('max') - df.timestamp).dt.seconds
    df['phase_elapse_start'] = (
            df.timestamp - df.groupby(['process_id', 'return_phase']).timestamp.transform('min')).dt.seconds
    df['end_turb'] = df.return_turbidity * (df.phase_elapse_end <= 40)  # Last 80 seconds, since each record = 2 seconds
    df['end_residue'] = df.return_residue * (df.phase_elapse_end <= 40)

    logger.info('Pre-processing of raw ' + data_type + ' data finished.')

    if return_phase_defs is None:
        return df, return_phases, supply_phases
    else:
        return df


def calculate_start_times(df):
    """Calculates start times and days from start of dataset for each process."""

    output = pd.DataFrame(df.groupby(['process_id']).timestamp.min()).reset_index()
    output.timestamp = output.timestamp.astype('datetime64[ns]')
    output['day_number'] = output.timestamp.dt.dayofyear - 51
    output.columns = ['process_id', 'start_time', 'day_number']

    return output
