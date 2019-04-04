# -*- coding: utf-8 -*-
import sys
sys.path.append(".")
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.build_features import calc_features
from src.common import ensure_dir

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from %s', input_filepath)

    train_labels = pd.read_hdf(input_filepath, "train_labels")
    recipe_metadata = pd.read_hdf(input_filepath, "recipe_metadata")
    train_test_events = pd.read_hdf(input_filepath, "train_test_events")

    logging.info("calculating features for train_test")
    train_test = calc_features(train_test_events, train_labels, recipe_metadata)

    logging.info("writing output to %s", output_filepath)
    ensure_dir(output_filepath)
    train_test.to_hdf(output_filepath, "train_test", mode="w")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
