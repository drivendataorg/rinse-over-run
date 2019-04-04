# -*- coding: utf-8 -*-
import click
import logging
from pathlib2 import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    training_values = pd.read_csv(input_filepath + 'train_values.csv', index_col=0,
                                  parse_dates=['timestamp'])
    testing_values = pd.read_csv(input_filepath + 'test_values.csv', index_col=0,
                                 parse_dates=['timestamp'])

    training_labels = pd.read_csv(
        input_filepath + 'train_labels.csv', index_col=0)
    cleaning_recipe = pd.read_csv(
        input_filepath + 'recipe_metadata.csv', index_col=0)
    cleaning_recipe_train = cleaning_recipe.loc[training_values.process_id.unique(
    )]
    cleaning_recipe_train['combination'] = cleaning_recipe_train['pre_rinse'].astype(str) + cleaning_recipe_train['caustic'].astype(
        str) + cleaning_recipe_train['intermediate_rinse'].astype(str) + cleaning_recipe_train['acid'].astype(str)

    training_values['supply_flow'] = np.maximum(
        training_values['supply_flow'], 0)
    training_values['return_turbidity'] = np.maximum(
        training_values['return_turbidity'], 0)
    training_values['return_flow'] = np.maximum(
        training_values['return_flow'], 0)
    training_values['supply_pressure'] = np.maximum(
        training_values['supply_pressure'], 0)
    training_values = training_values[training_values.phase != 'final_rinse']

    testing_values['supply_flow'] = np.maximum(
        testing_values['supply_flow'], 0)
    testing_values['return_turbidity'] = np.maximum(
        testing_values['return_turbidity'], 0)
    testing_values['return_flow'] = np.maximum(
        testing_values['return_flow'], 0)
    testing_values['supply_pressure'] = np.maximum(
        testing_values['supply_pressure'], 0)

    train_data = training_values.copy()
    comb_train = cleaning_recipe_train.loc[train_data.process_id.unique()]
    ffff = comb_train[comb_train.combination == '1111']
    ffoo = comb_train[comb_train.combination == '1100']
    foof = comb_train[comb_train.combination == '1001']

    train_data = train_data[~((train_data.process_id.isin(
        foof)) & (train_data.phase == 'pre_rinse'))]

    rng1 = np.random.RandomState(181)
    not_to_keep1 = rng1.choice(
        ffoo.index.tolist(),
        size=np.int(len(ffoo) * 0.05),
        replace=False)

    not_to_keep1_3 = rng1.choice(
        ffoo.index.tolist(),
        size=np.int(len(ffoo) * 0.01),
        replace=False)
    train_data = train_data[~((train_data.process_id.isin(
        not_to_keep1)) & (train_data.phase == 'caustic'))]
    train_data = train_data[~((train_data.process_id.isin(
        not_to_keep1_3)) & (train_data.phase == 'pre_rinse'))]

    rng = np.random.RandomState(181)
    not_to_keep = rng.choice(
        ffff.index.tolist(),
        size=np.int(len(ffff) * 0.6),
        replace=False)

    not_to_keep_40 = rng.choice(
        ffff.index.tolist(),
        size=np.int(len(ffff) * 0.3),
        replace=False)

    not_to_keep_10 = rng.choice(
        ffff.index.tolist(),
        size=np.int(len(ffff) * 0.08),
        replace=False)

    not_to_keep_1 = rng.choice(
        ffff.index.tolist(),
        size=np.int(len(ffff) * 0.01),
        replace=False)

    train_data = train_data[~((train_data.process_id.isin(
        not_to_keep)) & (train_data.phase == 'acid'))]
    train_data = train_data[~((train_data.process_id.isin(not_to_keep_40)) & (
        train_data.phase == 'intermediate_rinse'))]
    train_data = train_data[~((train_data.process_id.isin(
        not_to_keep_10)) & (train_data.phase == 'caustic'))]
    train_data = train_data[~((train_data.process_id.isin(
        not_to_keep_1)) & (train_data.phase == 'pre_rinse'))]

    train_data.to_pickle(output_filepath + 'train_data.p')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(dotenv_path=find_dotenv(), verbose=True)

    main()
