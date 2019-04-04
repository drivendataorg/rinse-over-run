# -*- coding: utf-8 -*-
import click
import logging
from pathlib2 import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from src.features.build_features import *
import cPickle as pickle


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    training_values = pd.read_pickle(input_filepath + 'train_data.p')
    training_values = training_values.reset_index()
    training_values = training_values.drop('row_id', axis=1)
    training_values['supply_flow'] = np.maximum(
        training_values['supply_flow'], 0)
    training_values['supply_pressure'] = np.maximum(
        training_values['supply_pressure'], 0)
    training_values['return_turbidity'] = np.maximum(
        training_values['return_turbidity'], 0)
    training_values['return_flow'] = np.maximum(
        training_values['return_flow'], 0)
    training_values['conductivity'] = training_values['return_conductivity'] / \
        (1.0 + (1.5 / 100 * (training_values['return_temperature'] - 25)))
    training_values['conductivity'] = np.maximum(
        training_values['conductivity'], 0)

    testing_values = pd.read_csv(
        input_filepath + '../raw/test_values.csv', index_col=0, parse_dates=['timestamp'])
    testing_values['supply_flow'] = np.maximum(
        testing_values['supply_flow'], 0)
    testing_values['supply_pressure'] = np.maximum(
        testing_values['supply_pressure'], 0)
    testing_values['return_turbidity'] = np.maximum(
        testing_values['return_turbidity'], 0)
    testing_values['return_flow'] = np.maximum(
        testing_values['return_flow'], 0)
    testing_values['conductivity'] = testing_values['return_conductivity'] / \
        (1.0 + (1.5 / 100 * (testing_values['return_temperature'] - 25)))
    testing_values['conductivity'] = np.maximum(
        testing_values['conductivity'], 0)

    for i in [5, 10, 50, 100, 200, 500]:
        df = prep_time_series_by_days(training_values, i)
        print 'ts_%d done...' % i
        df_test = prep_time_series_by_days(testing_values, i)
        print 'ts_%d done...' % i
        df.to_csv(output_filepath + 'CSV/train_ts_%d.csv' % i)
        df_test.to_csv(output_filepath + 'CSV/test_ts_%d.csv' % i)

        with open(output_filepath + 'pickle/train_ts_%d.p' % i, 'wb') as fdf:
            pickle.dump(df, fdf)

        with open(output_filepath + 'pickle/test_ts_%d.p' % i, 'wb') as fdf:
            pickle.dump(df_test, fdf)

    df = get_operating_time(training_values)
    print 'Operating Time Done...'
    df_test = get_operating_time(testing_values)
    print 'Operating Time test done...'

    df.to_csv(output_filepath + 'CSV/train_operating.csv')
    df_test.to_csv(output_filepath + 'CSV/test_operating.csv')

    with open(output_filepath + 'pickle/train_operating.p', 'wb') as fdf:
        pickle.dump(df, fdf)

    with open(output_filepath + 'pickle/test_operating.p', 'wb') as fdf:
        pickle.dump(df_test, fdf)

    df = prep_boolean(training_values)
    print 'Boolean done...'
    df_test = prep_boolean(testing_values)
    print 'Boolean test done...'

    df.to_csv(output_filepath + 'CSV/train_boolean.csv')
    df_test.to_csv(output_filepath + 'CSV/test_boolean.csv')

    with open(output_filepath + 'pickle/train_boolean.p', 'wb') as fdf:
        pickle.dump(df, fdf)

    with open(output_filepath + 'pickle/test_boolean.p', 'wb') as fdf:
        pickle.dump(df_test, fdf)

    df, df_test = prep_categorical(training_values, testing_values)
    print 'categorical done...'

    df.to_csv(output_filepath + 'CSV/train_categorical.csv')
    df_test.to_csv(output_filepath + 'CSV/test_categorical.csv')

    with open(output_filepath + 'pickle/train_categorical.p', 'wb') as fdf:
        pickle.dump(df, fdf)

    with open(output_filepath + 'pickle/test_categorical.p', 'wb') as fdf:
        pickle.dump(df_test, fdf)

    df = prep_time_series_features(training_values)
    print 'ts done...'
    df_test = prep_time_series_features(testing_values)
    print 'ts test done...'

    df.to_csv(output_filepath + 'CSV/train_ts_all.csv')
    df_test.to_csv(output_filepath + 'CSV/test_ts_all.csv')

    with open(output_filepath + 'pickle/train_ts_all.p', 'wb') as fdf:
        pickle.dump(df, fdf)

    with open(output_filepath + 'pickle/test_ts_all.p', 'wb') as fdf:
        pickle.dump(df_test, fdf)

    df = phase_data(training_values)
    print 'phase_data done...'
    df_test = phase_data(testing_values)
    print 'phase_data test done...'

    df.to_csv(output_filepath + 'CSV/train_phase.csv')
    df_test.to_csv(output_filepath + 'CSV/test_phase.csv')

    with open(output_filepath + 'pickle/train_phase.p', 'wb') as fdf:
        pickle.dump(df, fdf)

    with open(output_filepath + 'pickle/test_phase.p', 'wb') as fdf:
        pickle.dump(df_test, fdf)

    df = prep_spent_time(training_values)
    print 'spent_time done...'
    df_test = prep_spent_time(testing_values)
    print 'spent_time test done...'

    df.to_csv(output_filepath + 'CSV/train_time.csv')
    df_test.to_csv(output_filepath + 'CSV/test_time.csv')

    with open(output_filepath + 'pickle/train_time.p', 'wb') as fdf:
        pickle.dump(df, fdf)

    with open(output_filepath + 'pickle/test_time.p', 'wb') as fdf:
        pickle.dump(df_test, fdf)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(dotenv_path=find_dotenv(), verbose=True)

    main()
