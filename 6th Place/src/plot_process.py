#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.common import ensure_dir
from src.visualization.visualize import plot_process

@click.command()
@click.option('--input-filepath', type=click.Path(exists=True), default='data/interim/train_test.hdf5')
@click.option('--output-filepath', type=click.Path(), default='/tmp/a.png')
@click.option("--process-id", type=int)
@click.option("--object-id", type=int)
@click.option("--dpi", type=int, default=150)
def main(input_filepath, output_filepath, process_id, object_id, dpi):
    logger = logging.getLogger(__name__)
    logger.info('loading data from %s', input_filepath)

    train_test_events = pd.read_hdf(input_filepath, "train_test_events")

    if process_id is not None:
        logging.info("plotting process_id=%d", process_id)
        g = train_test_events[train_test_events.process_id==process_id]
        object_id = g.iloc[0].object_id
        fig, _ = plot_process(g)
        fn = output_filepath
        ensure_dir(fn)
        fig.savefig(fn, dpi=dpi, bbox_inches='tight')
        # plt.close()
    elif object_id is not None:
        grp = train_test_events[train_test_events.object_id==object_id].groupby('process_id')
        for process_id, g in grp:
            logging.info("plotting process_id=%d", process_id)
            fig, _ = plot_process(g)
            fn = os.path.join(output_filepath, f"plt_obj{object_id}_pr{process_id}.png")
            ensure_dir(fn)
            fig.savefig(fn, dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        g = [(process_id, train_test_events[train_test_events.process_id==process_id])]
        fig, _ = plot_process(g)
        fig.savefig(output_filepath, dpi=dpi, bbox_inches='tight')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
