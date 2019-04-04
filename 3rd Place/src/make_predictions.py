import pandas as pd
import os
import sys
from datetime import datetime
import logging as logger

sys.path.append(os.getcwd() + '\\src\\')  # Add path to src folder containing other python modules

from engineer_features import create_model_datasets
from build_models import build_test_models


def predict_test_values(raw_data, test_data, start_times, metadata, path, params, response, test_iterations, labels,
                        cols_to_include, save_to_local=False):
    """Builds models on full training test and makes predictions on test set.

    Predictions are saved locally in the 'Predictions' directory. The names of the local files include timestamps,
        so they will not be overwritten by later predictions.

    Args:
        raw_data (dataframe): full training dataset.
        test_data (dataframe): test dataset.
        start_times (dataframe): start times for each process.
        metadata (dataframe): recipe metadata dataset.
        path (str): path to directory containing source files.
        params (dict): four keys corresponding to four model types, and associated values are dictionaries containing
            lightgbm model parameters for that model type (e.g. {'boosting_type': 'gbdt', 'num_leaves': 63, ...}).
        response (str): name of response column. must be present in both dataframes passed through
            'raw_data' and 'test_data' args.
        test_iterations (dict): dictionary with four keys, one for each model type, and corresponding values
            indicate best number of iterations when training the model on full training data.
        labels (dataframe): response values for each process in the training data
        cols_to_include (dict): entire dictionary of columns to include for each of the four models.
            Note that this arg is not the same as the cols_to_include arg in many other functions!
        save_to_local (bool): flag to indicate whether or not to save the model-ready dataset to local directory
            specified by 'path' arg.

    Returns:
        N/A.
    """

    # Initialize list of predictions; will contain four dataframes corresponding to predictions from the four models
    test_predictions = []

    # Create model-ready datasets from the full training data and test data
    processed_full_train_data, \
        processed_test_data = create_model_datasets(raw_data, test_data, start_times, labels, response, metadata,
                                                    path, val_or_test='test', save_to_local=save_to_local)

    # Build the four test models and make the predictions on the set
    for model_type in cols_to_include.keys():
        test_predictions = build_test_models(model_type, processed_full_train_data, processed_test_data,
                                             response, params[model_type], test_iterations,
                                             cols_to_include[model_type], test_predictions)

    # Combine predictions from four models into one dataframe
    test_predictions = pd.concat(test_predictions).sort_values(by='process_id')

    # Handle negative values by setting them equal to the lowest predicted value
    # Negative predictions occurred on rare occasions early on, should not happen with most current modeling approach
    test_predictions.loc[test_predictions[response] < 0, response] = \
        test_predictions.loc[test_predictions[response] > 0, response].min()

    # Save results to csv
    write_predictions_to_csv(test_predictions)


def write_predictions_to_csv(predictions):
    """Saves test set predictions to csv file in 'Predictions' directory."""

    current_time = str(datetime.now().replace(microsecond=0)).replace(':', '.')
    current_directory = os.getcwd()

    # Predictions sorted by process_id - these are submitted to leaderboard
    output_path = current_directory + '\\predictions\\Test Predictions ' + current_time + '.csv'
    predictions.to_csv(output_path, index=False)

    logger.info('Test set predictions made at ' + str(current_time) + ' saved to csv file.')
