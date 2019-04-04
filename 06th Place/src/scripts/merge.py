#!/usr/bin/env python
import sys
sys.path.append(".")
import logging
import click
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#


@click.command()
@click.option("-i", "--input", type=click.Path(exists=True), multiple=True)
@click.option("-o", "--output", type=click.Path())
def main(input, output):
    logging.info(f"blend input={input} output={output}")
    df_inp = [
        pd.read_csv(fn, index_col='process_id')['final_rinse_total_turbidity_liter'].rename(i)
        for i, fn in enumerate(input, start=1)
    ]
    df = pd.concat(df_inp, axis=1)
    print(df.head(10))
    blended = df.ffill(axis=1).iloc[:, -1]
    print(blended.head(10))

    submission = pd.read_csv("data/raw/submission_format.csv")
    submission = submission.join(blended.rename('new'), how='left', on='process_id')
    submission['final_rinse_total_turbidity_liter'] = submission['new']
    del submission['new']
    assert all(submission['final_rinse_total_turbidity_liter'].notnull())

    now = pd.Timestamp.now()
    fn = output if output else f"/tmp/submission-{now.strftime('%Y%m%d-%H%M')}.csv"
    logging.info(f"writing submission to {fn}")
    submission.to_csv(fn, index=False)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_fmt)
    main() 
