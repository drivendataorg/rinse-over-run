import pandas as pd
import logging as logger
import re


def create_model_datasets(df_train, df_test, start_times, labels, response, metadata, path,
                          val_or_test='validation', save_to_local=False):
    """Transforms pre-processed train and test data into model-ready dataframes.

    Performs following steps:
    1. Creates and merges normalization tables used for feature engineering.
    2. Engineers features at a process level or more granular (i.e. return-phase, supply-phase, etc.)
    3. Joins labels and metadata to engineered features
    4. Removes outliers
    5. Converts appropriate features to categorical

    Args:
        df_train (dataframe): training dataset.
        df_test (dataframe): test dataset.
        start_times (dataframe): start times for each process.
        labels (dataframe): labels for training dataset.
        response (str): name of response column.
        metadata (dataframe): recipe metadata dataset.
        path (str): path to directory containing source files.
        val_or_test (str): flag to indicate whether or not the test data being passed is a validation set or a true
            test set. Valid values are 'validation' and 'test'.
        save_to_local (bool): flag to indicate whether or not to save the model-ready dataset to local directory
            specified by 'path' arg.

    Returns:
        processed_train_data (dataframe): fully processed train dataset for modeling.
        processed_val_data (dataframe): fully processed validation or test data set for making predictions and
            evaluating model performance (for validation sets only).
    """

    # Create normalization lookup tables
    logger.info('Creating and merging normalization lookup tables...')

    train_lookup = df_train.groupby(['object_id', 'return_phase']).\
        agg({'return_flow': 'median', 'return_conductivity': 'median'}).reset_index()
    train_lookup.columns = ['object_id', 'return_phase', 'median_return_flow', 'median_conductivity']
    df_train = df_train.merge(train_lookup, on=['object_id', 'return_phase'])
    df_test = df_test.merge(train_lookup, on=['object_id', 'return_phase'])

    train_lookup = df_train.groupby(['object_id', 'supply_phase']).\
        agg({'supply_flow': 'median', 'supply_pressure': 'median'}).reset_index()
    train_lookup.columns = ['object_id', 'supply_phase', 'median_supply_flow', 'median_supply_pressure']
    df_train = df_train.merge(train_lookup, on=['object_id', 'supply_phase'], how='left').sort_values(by='timestamp')
    df_test = df_test.merge(train_lookup, on=['object_id', 'supply_phase'], how='left').sort_values(by='timestamp')

    logger.info('Normalization lookup tables finished.')

    # Engineer phase-level features on train, validation, and test sets
    logger.info('Engineering features on train, ' + val_or_test + ' sets...')
    processed_train_data = engineer_features(df_train, start_times)
    processed_val_data = engineer_features(df_test, start_times)
    logger.info('Successfully engineered features.')

    # Fill nas with 0, where appropriate
    for model_type in ['acid', 'int_rinse', 'pre_rinse', 'caustic']:
        cols = list(filter(lambda x: re.search(r'(?=.*' + model_type + ')', x), list(processed_train_data.columns)))
        processed_train_data.loc[pd.notna(processed_train_data['row_count_' + model_type]), cols] =\
            processed_train_data.loc[pd.notna(processed_train_data['row_count_' + model_type]), cols].fillna(0)
        # processed_train_data.loc[:, cols] = processed_train_data.loc[:, cols].fillna(-1)

    # Drop features that make no sense (produce mostly 0 or nan)
    keep_cols = processed_val_data.apply(lambda x: (x.isnull()).sum() / len(x)) <= 0.9
    processed_train_data = processed_train_data[list(keep_cols[keep_cols].index)]
    processed_val_data = processed_val_data[list(keep_cols[keep_cols].index)]

    # Bring in labels and metadata for train and validation data
    processed_train_data = processed_train_data.merge(labels, on='process_id').merge(metadata, on='process_id')
    processed_val_data = processed_val_data.merge(metadata, on='process_id')
    if val_or_test == 'validation':
        processed_val_data = processed_val_data.merge(labels, on='process_id')

    # Remove outliers from training data
    processed_train_data = remove_outliers(processed_train_data, response)

    # Write datasets out to local, if desired
    if val_or_test == 'validation' and save_to_local:
        processed_train_data.to_csv(path + 'modeling_data_train.csv')
        logger.info('Training modeling data successfully saved to csv file.')
        processed_val_data.to_csv(path + 'modeling_data_validation.csv')
        logger.info('Validation modeling data successfully saved to csv file.')
    elif val_or_test == 'test' and save_to_local:
        processed_train_data.to_csv(path + 'modeling_data_full.csv')
        logger.info('Full training modeling data successfully saved to csv file.')
        processed_val_data.to_csv(path + 'modeling_data_test.csv')
        logger.info('Test modeling data successfully saved to csv file.')
    elif not save_to_local:
        pass
    else:
        logger.error('Invalid value for val_or_test or save_to_local args; val_or_test must be ' +
                     '\'validation\' or \'test\' and save_to_local must be boolean (True or False).')

    # Convert object id to category
    # Ensure that categories are consistent across training, validation, and test sets
    for col in ['object_id', 'recipe_type']:
        processed_train_data[col] = processed_train_data[col].astype('category')
        processed_val_data[col] = processed_val_data[col].astype(pd.api.types.CategoricalDtype(
            categories=processed_train_data[col].cat.categories))

    processed_val_data = processed_val_data.sort_values(by='process_id')

    return processed_train_data, processed_val_data


def engineer_features(df, timestamps):
    """Engineers features from raw data at various levels of aggregation and transforms them to process-level.

    Args:
        df (dataframe): preprocessed process-timestamp-level data.
        timestamps (dataframe): start times for each process.

    Returns:
        df_final_output (dataframe): full set of engineered process-level features for use in modeling.
    """

    # Normalize flows
    df['norm_return_flow'] = df.return_flow / df.median_return_flow
    df['norm_turb'] = df.norm_return_flow * df.return_turbidity

    df['norm_supply_pressure'] = df.supply_pressure - df.median_supply_pressure
    df['norm_conductivity'] = df.return_conductivity - df.median_conductivity

    group_cols = ['process_id', 'object_id', 'pipeline']

    # Calculate features at various levels of aggregation
    df_return_phase = calculate_features(df, group_cols, 'return_phase')
    df_supply_phase = calculate_features(df, group_cols, 'supply_phase', df_return_phase)
    df_full_phase = calculate_features(df, group_cols, 'phase', df_supply_phase)
    df_final_output = calculate_features(df, group_cols, 'process', df_full_phase)

    # Bring in start times for processed data
    df_final_output = df_final_output.merge(timestamps, on='process_id')
    df_final_output = df_final_output.sort_values(by=['object_id', 'start_time'])

    return df_final_output


def calculate_features(df, base_group_cols, level='process', existing_features=None):
    """Calculates values of engineered features.

    This function contains the primary code for feature engineering. Features can only be created at one level of
        aggregation per function call. In other words, all return-phase level features can be created with one call,
        then supply-phase features can be created and merged with the existing return-phase features in a second call
        (using the 'existing_features' parameter), and so on.
    Adding new features at any level of aggregation (i.e. not process-timestamp) should be done here.

    Args:
        df (dataframe): dataframe of process-timestamp-level data to have features engineered from.
        base_group_cols (list of str): broadest level of aggregation permitted. In this case, it is process-level
            plus any other categorical variables which we effectively want to exclude from aggregation
            (object, pipeline). Only column names in 'df' arg are valid values in this list.
        level (str): Additional columns to group by when engineering aggregated features, on top of those specified
            in 'base_group_cols' arg.
        existing_features (dataframe): existing feature dataset. The features created during the function call will
            be merged with the existing feature set before the output is returned by the function.

    Returns:
        output (dataframe): newly engineered feature set, merged with existing features if they were passed through
            the 'existing_features' arg.
    """

    # Determine true level of aggregation for set of features to be engineered
    full_group_cols = base_group_cols + [level] if level != 'process' else base_group_cols
    df_groupby = df.groupby(full_group_cols)

    if level == 'return_phase':
        features = pd.DataFrame({'ret_turb': df_groupby.norm_turb.sum(),
                                 'ret_residue': df_groupby.return_residue.sum(),
                                 'ret_cond': df_groupby.norm_conductivity.min(),
                                 'ret_dur': (df_groupby.timestamp.max() -
                                             df_groupby.timestamp.min()).astype('timedelta64[s]')
                                 }).reset_index()
    elif level == 'supply_phase':
        features = pd.DataFrame({'sup_flow': df_groupby.supply_flow.sum(),
                                 'sup_press': df_groupby.norm_supply_pressure.min(),
                                 'sup_dur': (df_groupby.timestamp.max() -
                                             df_groupby.timestamp.min()).astype('timedelta64[s]'),
                                 }).reset_index()
    elif level == 'phase':
        features = pd.DataFrame({'row_count': df_groupby.phase.count(),

                                 'end_turb': df_groupby.end_turb.mean(),
                                 'end_residue': df_groupby.end_residue.sum(),

                                 'return_temp': df_groupby.return_temperature.min(),

                                 'lsh_caus': df_groupby.tank_lsh_caustic.sum() /
                                             (df_groupby.timestamp.max() - df_groupby.timestamp.min()).astype(
                                              'timedelta64[s]'),
                                 'obj_low_lev': df_groupby.object_low_level.sum() /
                                                (df_groupby.timestamp.max() - df_groupby.timestamp.min()).astype(
                                                'timedelta64[s]'),
                                 }).reset_index()
    else:
        features = pd.DataFrame({'total_duration': (df_groupby.timestamp.max() -
                                                    df_groupby.timestamp.min()).astype('timedelta64[s]')
                                 }).reset_index()

    # If the features are at a more granular level than 'process', bring them up to process-level by unpivoting columns
    # For example, 'row_count' is calculated at a phase-level, so this will transform 'row_count' into
    #   'row_count_pre_rinse', 'row_count_caustic', etc.
    if level != 'process':
        features = pd.pivot_table(features,
                                  index=base_group_cols,
                                  columns=level,
                                  values=list(set(features.columns) - set(full_group_cols))).reset_index()

        features.columns = [' '.join(col).strip() for col in features.columns.values]
        features.columns = features.columns.str.replace(' ', '_')

    # Merge newly created process-level features to existing ones, if necessary
    output = features if existing_features is None else features.merge(existing_features, on=base_group_cols)

    return output


def remove_outliers(processed_train_data, response):
    """Remove outliers from dataset. Only training data should be passed to this function."""

    # Remove processed train data with unusually short or long train duration
    output = processed_train_data[(processed_train_data.total_duration > 30) &
                                  (processed_train_data.total_duration < 10000)]

    duration_dim = output.shape[0]
    logger.info('Number of duration outliers removed: ' + str(processed_train_data.shape[0] - duration_dim))

    # Remove response outliers with unusually low responses
    quantiles = (output.groupby('object_id')[response].quantile(0.5) / 25).reset_index()
    quantiles.columns = ['object_id', 'response_thresh']
    output = output.merge(quantiles, on='object_id')

    output = output[output.response_thresh < output[response]]
    logger.info('Number of response outliers removed: ' + str(duration_dim - output.shape[0]))

    return output
