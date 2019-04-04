# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(".")
from src.pandas_helpers import reduce_memory_usage
from src.common import ensure_dir


def load_data_from_file(filename, is_train=True):
    chunks = []
    for chunk in tqdm(pd.read_csv(filename, chunksize=10**5), f"{os.path.basename(filename)}"):
        chunks.append(chunk)
    df = pd.concat(chunks, axis=0, sort=False, ignore_index=True)
    df['is_train'] = is_train
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['pipeline'] = df['pipeline'].str[1:].map(int)
    df['phase_num'] = df['phase'].map({
        'pre_rinse': 1,
        'caustic': 2,
        'intermediate_rinse': 3, 
        'acid': 4, 
        'final_rinse': 5, 
    })

    def label(df, prefix, values):
        columns = [f"{prefix}_{v}" for v in values]
        mul = []
        d = {0: "none"}
        for i, v in enumerate(values):
            d[2**i] = v
            mul.append(2**i)
        return df[columns].applymap(int).multiply(mul, axis=1).sum(axis=1)\
            .map(d).fillna("unknown")

    logging.info("generating return_label")
    df['return_label'] = label(df, prefix='return', values=['caustic', 'acid', 'recovery_water','drain'])
    logging.info("generating supply_label")
    df['supply_label'] = label(df, prefix='supply', values=['pre_rinse', 'caustic', 'acid', 'clean_water'])

    sel = df[(df.phase == 'final_rinse') & (df.target_time_period)]
    df.loc[sel.index, 'phase_num'] = 6
    df['total_turbidity_liter'] = df['return_turbidity'] * df['return_flow'].clip(0, None)

    for label, groupby in [
        ('i', ['process_id']),
        ('p_i', ['process_id', 'phase_num'])
    ]:
        df[label] = df.groupby(groupby).cumcount() + 1
        df['rev_'+label] = df.groupby(groupby).cumcount(ascending=False) + 1

    df = reduce_memory_usage(df)
    logging.info("%s: loaded %d rows", os.path.join(filename), len(df))
    return df


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making interim data set from raw data input=%s', input_filepath)

    train_labels = pd.read_csv(os.path.join(input_filepath, "train_labels.csv"), index_col='process_id')
    recipe_metadata = pd.read_csv(os.path.join(input_filepath, "recipe_metadata.csv"), index_col='process_id')

    train_test_events = pd.concat([
        load_data_from_file(os.path.join(input_filepath, "train_values.zip"), is_train=True),
        load_data_from_file(os.path.join(input_filepath, "test_values.zip"), is_train=False),
    ], axis=0, ignore_index=True, sort=False)

    logging.info("writing output to %s", output_filepath)
    ensure_dir(output_filepath)
    train_labels.to_hdf(output_filepath, "train_labels", mode="w")
    recipe_metadata.to_hdf(output_filepath, "recipe_metadata", mode="a")
    train_test_events.to_hdf(output_filepath, "train_test_events", mode="a")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
