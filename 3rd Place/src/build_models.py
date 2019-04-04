import lightgbm as lgb
import pandas as pd
import numpy as np
import re
import sys
import os
import logging as logger

sys.path.append(os.getcwd() + '\\src\\')  # Add path to src folder containing other python modules

from visualize_insights import plot_shap


def subset_df_cols(regex, df):
    """Returns the subset of columns that match a particular regular expression."""

    return set(filter(lambda x: re.search(regex, x), list(df.columns)))


def select_model_columns(processed_train_data, cols_subset=None):
    """Creates a dictionary of columns to include in each of the four models.

    This function already contains a "base set' of columns to exclude from each model, derived from the inherent nature
        of the problem and previous feature selection runs. For example, all predictors derived from the caustic phase
        should be excluded from the pre_rinse from, which is why the associated regex includes '.*caustic|'.

    Args:
        processed_train_data (dataframe): training data.
        cols_subset (str): regex for additional columns to exclude when building models.
            Primarily used when doing grid search feature selection.

    Returns:
        cols_to_include (dict): keys are the four model types, values are lists of columns to keep when building
            the model type given by the corresponding key.
    """

    # For each of the four models, identify which columns should be kept from overall set
    # Simulates data censoring in test data
    pre_rinse_cols = subset_df_cols(r'(?=.*caustic|.*int_rinse|.*acid|.*other|.*residue|.*cond|.*temp)',
                                    processed_train_data)
    caustic_cols = subset_df_cols(r'(?=.*turb_pre_rinse|.*int_rinse|.*acid|.*other|.*residue|.*cond|.*temp)',
                                  processed_train_data)
    int_rinse_cols = subset_df_cols(r'(?=.*turb_pre_rinse|.*acid|.*other|.*flow|.*residue|recipe.*)',
                                    processed_train_data)
    acid_cols = subset_df_cols(r'(?=turb_acid|.*turb_caustic|.*residue_pre_rinse|.*turb_pre_rinse|.*residue_int_rinse|.*turb_int_rinse|.*flow|.*sup|recipe.*)',
                               processed_train_data)

    base_cols = list(subset_df_cols(r'(?=.*row_count|total.*|.*none)', processed_train_data))
    misc_cols = ['response_thresh', 'day_number', 'start_time', 'process_id', 'pipeline']

    exclude_cols = set(misc_cols + base_cols) \
        if cols_subset is None \
        else set(list(subset_df_cols(r'(?=.*' + cols_subset + ')', processed_train_data)) + misc_cols + base_cols)
    all_cols = set(processed_train_data.columns)

    cols_to_include = {
        'pre_rinse': list(all_cols - exclude_cols - pre_rinse_cols),
        'caustic':   list(all_cols - exclude_cols - caustic_cols),
        'int_rinse': list(all_cols - exclude_cols - int_rinse_cols),
        'acid':      list(all_cols - exclude_cols - acid_cols)
    }

    return cols_to_include


def build_lgbm_validation_datasets(train_data, val_data, model_type, response, cols_to_include):
    """Builds training and validation lightgbm datasets.

    Args:
        train_data (dataframe): training data.
        val_data (dataframe): validation data.
        model_type (str): valid values are 'pre_rinse', 'caustic', 'int_rinse', and 'acid'.
        response (str): name of response column. must be present in both dataframes passed through 'train_data' and
            'val_data' args.
        cols_to_include (list of str): list of columns to include in output dataset.

    Returns:
        A dict with two keys, 'train' and 'eval'; the corresponding values are objects of class lightgbm.Dataset
            with the appropriate data for building lightgbm models.
    """

    # Only keep records in validation set that correspond to the model type being built
    # For example, some records do not have acid data, so we would not use an acid model to make predictions on them.
    val_data = val_data[val_data['row_count_' + model_type].notnull()]

    y_train = train_data.loc[:, response]
    y_val = val_data.loc[:, response]

    x_train = train_data[cols_to_include]
    x_val = val_data[cols_to_include]

    # Create lightgbm datasets for training and evaluation (validation)
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)

    return {'train': lgb_train,
            'eval': lgb_eval
            }


def build_lgbm_test_datasets(full_train_data, test_data, response, cols_to_include):
    """Builds training and test lightgbm datasets.

    Args:
        full_train_data (dataframe): full training data.
        test_data (dataframe): test data.
        response (str): name of response column. must be present in both dataframes passed through
            'full_train_data' and 'test_data' args.
        cols_to_include (list of str): list of columns to include in output dataset.

    Returns:
        A dict with five keys, 'full_train' and 'test_phase_x', where 'x' corresponds to the four model types.
        The corresponding values are objects of class lightgbm.Dataset with the appropriate data for building
            lightgbm models.
        """

    cols_to_include = cols_to_include + ['process_id']

    # Split the full test data into four parts; each part should contains the processes whose responses will be
    #   predicted by the associated model.
    test_data_acid = test_data[test_data.row_count_acid.notnull()]
    test_data_int_rinse = test_data[test_data.row_count_acid.isnull() &
                                    test_data.row_count_int_rinse.notnull()]
    test_data_caustic = test_data[test_data.row_count_acid.isnull() &
                                  test_data.row_count_int_rinse.isnull() &
                                  test_data.row_count_caustic.notnull()]
    test_data_pre_rinse = test_data[test_data.row_count_acid.isnull() &
                                    test_data.row_count_int_rinse.isnull() &
                                    test_data.row_count_caustic.isnull()]

    y_train = full_train_data.ix[:, response]

    x_test_acid = test_data_acid[cols_to_include]
    x_test_pre_rinse = test_data_pre_rinse[cols_to_include]
    x_test_caustic = test_data_caustic[cols_to_include]
    x_test_int_rinse = test_data_int_rinse[cols_to_include]

    if 'process_id' in cols_to_include:
        cols_to_include.remove('process_id')
    x_train = full_train_data[cols_to_include]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)

    return {'full_train': lgb_train,
            'test_acid': x_test_acid,
            'test_pre_rinse': x_test_pre_rinse,
            'test_caustic': x_test_caustic,
            'test_int_rinse': x_test_int_rinse
            }


def build_validation_models(model_type, processed_train_data, processed_val_data, params, response, cols_to_include,
                            train_ratio, max_train_ratio, tuning_params, validation_results, cols,
                            predictions=None, visualize=False, vis_dependence_predictor=None, vis_cutoff=None,
                            vis_interaction='auto'):
    """Builds LightGBM datasets and models for train/validation purposes.

    Args:
        model_type (str): valid values are 'pre_rinse', 'caustic', 'int_rinse', and 'acid'.
        processed_train_data (dataframe): model-ready training data at a process level.
        processed_val_data (dataframe): model-ready validation data at a process level.
        params (dict): four keys corresponding to four model types, and associated values are dictionaries containing
            lightgbm model parameters for that model type (e.g. {'boosting_type': 'gbdt', 'num_leaves': 63, ...}).
        response (str): name of response column. must be present in both dataframes passed through
            'processed_train_data' and 'processed_val_data' args.
        cols_to_include (list of str): list of columns to include in lightgbm datasets.
        train_ratio (int): number of days of data to include in training set. Valid range is from 1 to 63.
        max_train_ratio (int): maximum number of days to include in training set across all train/validation splits.
        tuning_params (tuple): values of parameters that were tuned using grid search.
        validation_results (dataframe): performance results for each model that was built. Each time this function
            is called, a row will be appended to this dataframe with the results of the model built.
        cols (str): list of columns excluded from model training as part of grid search feature selection.
        predictions (dataframe): validation set predictions made to evaluate model performance. Must either be a
            dataframe of prior validation set predictions, in which case the validation set predictions made by this
            model will be appended to the previous ones, or None, in which case no predictions are returned.
        visualize (bool): whether or not to display SHAP plots. If True, SHAP plots will be generated for the acid
            model created on the largest training data set specified by the 'max_train_ratio' arg.
        vis_dependence_predictor (str): specifies which predictor to make a dependence plot for. Must be the name of
            a predictor in the acid model. Will only generate a plot if 'visualize' arg is True.
        vis_cutoff (numeric): x-axis cutoff value for SHAP dependence plot. Only has an effect if 'visualize' arg
            is True and vis_dependence_predictor is a valid predictor name.
        vis_interaction (str): passed to the 'interaction_index' parameter for SHAP dependence plots. The value of
            'auto' will automatically choose a second predictor that SHAP estimates has the strongest interaction with
            the primary predictor specified by the 'vis_dependence_predictor' arg and illustrate the interaction using
            color, while None will not show any interactions and all data points will be blue.

    Returns:
        predictions (dataframe): validation set predictions, or None if the original value of the 'predictions' arg
            was None.
        validation_results (dataframe): the dataframe passed through the original validation_results arg, with one
            additional row appended that contains the performance data for the model built by this function call.
    """

    # Build lightgbm datasets from train and test data
    # Must be repeated for each model to properly simulate data censoring ('cols_to_include' parameter)
    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, model_type, response,
                                                   cols_to_include=cols_to_include)

    # Train model
    logger.info('Training ' + model_type + ' model...')
    gbm_train = lgb.train(params,
                          modeling_data['train'],
                          num_boost_round=5000,  # should be set very large in conjunction with low learning rate
                          valid_sets=modeling_data['eval'],
                          verbose_eval=False,
                          early_stopping_rounds=150,  # early stopping to prevent overfitting
                          keep_training_booster=True  # save validation set predictions
                          )

    # Save validation set predictions, if desired
    # Generally want to avoid this when doing grid search feature selection or hyperparameter tuning
    if predictions is not None:
        preds = gbm_train._Booster__inner_predict(data_idx=1)  # validation set predictions saved by lgb.train

        output_preds = processed_val_data[processed_val_data['row_count_' + model_type].notnull()].reset_index()
        output_preds = output_preds[['process_id', response]]
        output_preds['train_ratio'] = train_ratio
        output_preds['model_type'] = model_type
        output_preds['predicted_response'] = pd.Series(preds)

        predictions = predictions.append(output_preds)

    # Not sure why this needs to happen again, but SHAP visuals break if omitted
    modeling_data = build_lgbm_validation_datasets(processed_train_data, processed_val_data, model_type, response,
                                                   cols_to_include=cols_to_include)

    # Create SHAP plots, if desired
    if train_ratio == max_train_ratio and visualize is True and model_type == 'acid':
        plot_shap(gbm_train, modeling_data, vis_dependence_predictor, vis_interaction, vis_cutoff)

    # Append validation results for current model to those from previous models, train/validation splits
    validation_results = validation_results.append(pd.DataFrame([[model_type,
                                                                  train_ratio,
                                                                  cols,
                                                                  tuning_params[0],
                                                                  tuning_params[1],
                                                                  str(tuning_params[2]),
                                                                  round(gbm_train.best_score['valid_0']['mape'], 5),
                                                                  gbm_train.best_iteration]],
                                                   columns=validation_results.columns))

    return predictions, validation_results


def build_test_models(model_type, processed_full_train_data, processed_test_data, response, params, test_iterations,
                      cols_to_include, test_predictions):
    """Builds LightGBM datasets and models on full train/test data. Also makes predictions on test data.

    Args:
        model_type (str): valid values are 'pre_rinse', 'caustic', 'int_rinse', and 'acid'.
        processed_full_train_data (dataframe): model-ready full training data at a process level.
        processed_test_data (dataframe): model-ready test data at a process level.
        params (dict): four keys corresponding to four model types, and associated values are dictionaries containing
            lightgbm model parameters for that model type (e.g. {'boosting_type': 'gbdt', 'num_leaves': 63, ...}).
        response (str): name of response column. must be present in both dataframes passed through
            'processed_train_data' and 'processed_val_data' args.
        test_iterations (dict): four keys corresponding to four model types, and associated values are integers which
            specify how many rounds to train that particular model for (should be derived from validation set results).
        cols_to_include (list of str): list of columns to include in lightgbm datasets.
        test_predictions (list of dataframe): existing predictions made on test data by previous models built on
            full training data. Each of the elements in the list corresponds to a set of predictions made by a model on
            the appropriate subset of test data.

    Returns:
        test_predictions (dataframe): the same list of dataframes passed through the original 'test_predictions' arg,
            with one additional dataframe appended to the list corresponding to the predictions made during this
            function call.
    """

    # Build lgbm data sets on full train and test data
    prediction_data = build_lgbm_test_datasets(processed_full_train_data, processed_test_data, response,
                                               cols_to_include=cols_to_include)

    # Build model on full training data to make predictions for test set
    logger.info('Building model on full training data for ' + model_type + ' model...')

    gbm_full = lgb.train(params,
                         prediction_data['full_train'],
                         num_boost_round=test_iterations[model_type])

    # Make predictions on test set and save to .csv
    logger.info('Making test set predictions for ' + model_type + ' model...')

    test_predictions.append(pd.DataFrame({'process_id': prediction_data['test_' + model_type].process_id,
                                          response: gbm_full.predict(prediction_data['test_' + model_type])}
                                         ))

    return test_predictions


def calculate_validation_metrics(validation_results, save_to_local=False, path=None, verbose=False):
    """Calculates estimated test set error using validation set results and best number of iterations for each model.

    Args:
        validation_results (dataframe): validation set errors calculated by lightgbm on each model.
        save_to_local (bool): flag which indicates whether or not to save validation results to local file.
        path (str): path to directory where validation results will be saved, if desired.
        verbose (bool): flag which indicates whether or not to print model-specific details to console.

    Returns:
        test_iterations (dict): dictionary with four keys, one for each model type, and corresponding values
            indicate best number of iterations when training the model on full training data.
    """

    test_iterations = {}
    est_test_errors = {}
    phases = ['pre_rinse', 'caustic', 'int_rinse', 'acid']

    validation_results.Best_Num_Iters = validation_results.Best_Num_Iters.astype(int)

    validation_groupby = ['Model_Type', 'Num_Leaves', 'Excluded_Cols', 'Min_Data_In_Leaf', 'Min_Gain']
    validation_summary = validation_results.groupby(validation_groupby). \
        agg({'Best_MAPE': np.mean, 'Best_Num_Iters': np.median}).reset_index()
    validation_summary.Best_Num_Iters = validation_summary.Best_Num_Iters.astype(int)

    # Determine best hyperparameters for final model tuning
    validation_best = validation_summary.loc[validation_summary.groupby('Model_Type')['Best_MAPE'].idxmin()]

    # Create dictionaries for best test iterations and estimated errors for each of the four models
    for phase in phases:
        test_iterations[phase] = int(validation_best[validation_best.Model_Type == phase].Best_Num_Iters)
        est_test_errors[phase] = round(float(validation_best[validation_best.Model_Type == phase].Best_MAPE), 4)

    # Print validation result details to console, if desired
    if verbose:
        logger.info(validation_best)

        for phase in phases:
            logger.info('Best Iterations, ' + phase + ' model: ' + str(test_iterations[phase]) +
                        ('\n' if phase == 'acid' else ''))

        for phase in phases:
            logger.info('Estimated error for ' + phase + ' predictions: ' + str(est_test_errors[phase]) +
                        ('\n' if phase == 'acid' else ''))

    if save_to_local and path is not None:
        validation_results.to_csv(path + 'validation_results.csv')
        validation_summary.to_csv(path + 'validation_summary.csv')
        validation_best.to_csv(path + 'validation_best.csv')
    elif save_to_local and path is None:
        logger.error('Save to local flag is True, but path to save local files is missing.')
    elif not save_to_local:
        pass
    else:
        logger.error('Invalid value for save_to_local and/or path args.')

    logger.info('Estimated total error for all predictions: ' +
                str(round(292 / 2967 * est_test_errors['pre_rinse'] +
                          1205 / 2967 * est_test_errors['caustic'] +
                          672 / 2967 * est_test_errors['int_rinse'] +
                          798 / 2967 * est_test_errors['acid'], 4)))

    return test_iterations
