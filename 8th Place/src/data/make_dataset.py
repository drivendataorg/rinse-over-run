# -*- coding: utf-8 -*-

import click
import logging
import pandas as pd
import time
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from src.features.build_features import (
    Pipeline,
    InitTransformer,
    InputRecipeTransformer,
    ObservedRecipeTransformer,
    ObjectTransformer,
    PipelineTransformer,
    LabelTransformer,
    SimulationTransformer,
    AggTransformer,
    ColumnNameTransformer,
)


def read_csv(filepath_or_buffer, *args, **kwargs):
    start = time.time()
    ret = pd.read_csv(filepath_or_buffer, *args, **kwargs)
    duration = "%.2f" % (time.time() - start)
    memory = "%.2f" % (ret.memory_usage(deep=True).sum() / 1024 / 1024)
    logging.info(
        f"Loaded from {filepath_or_buffer} {len(ret)} rows, {len(ret.columns)} in {duration}s ({memory} Mb)"
    )
    return ret


def load_files(filepath):
    usecols = None
    tv = read_csv(
        filepath / "train_values.csv",
        # nrows=10000,
        index_col=0,
        parse_dates=["timestamp"],
        usecols=usecols,
    )
    tl = read_csv(filepath / "train_labels.csv", index_col=0)

    ev = read_csv(
        filepath / "test_values.csv",
        index_col=0,
        parse_dates=["timestamp"],
        usecols=usecols,
        # nrows=10000,
    )
    recipe_metadata = (
        read_csv(filepath / "recipe_metadata.csv")
        .set_index("process_id")
        .apply(
            lambda x: int("".join([str(it + 1) for it, i in enumerate(x) if i])), axis=1
        )
        .to_frame("input_recipe")
    )
    submission_format = read_csv(filepath / "submission_format.csv", index_col=0)
    return tv, tl, ev, recipe_metadata, submission_format


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)
    tv, tl, ev, recipe_metadata, submission_format = load_files(input_filepath)

    feature_encoder = Pipeline(
        [
            InitTransformer(),
            InputRecipeTransformer(recipe_metadata),
            ObservedRecipeTransformer(),
            ObjectTransformer(),
            PipelineTransformer(),
            LabelTransformer(labels=tl),
            SimulationTransformer(simulate=True),
            AggTransformer(
                funcnames=["max", "min", "mean", "median", "std", "sum"],
                cols=[
                    "supply_flow",
                    "supply_pressure",
                    "return_temperature",
                    "return_turbidity",
                    "return_flow",
                    "return_conductivity",
                    "tank_level_pre_rinse",
                    "tank_level_caustic",
                    "tank_level_acid",
                    "tank_level_clean_water",
                    "tank_temperature_pre_rinse",
                    "tank_temperature_caustic",
                    "tank_temperature_acid",
                    "tank_concentration_caustic",
                    "tank_concentration_acid",
                ],
            ),
            AggTransformer(
                funcnames=["mean", "sum"],
                cols=[
                    "return_drain",
                    "supply_pre_rinse",
                    "return_caustic",
                    "supply_caustic",
                    "return_acid",
                    "return_recovery_water",
                    "supply_acid",
                    "supply_clean_water",
                    "supply_pump",
                ],
            ),
            ColumnNameTransformer(),
        ]
    )
    feature_encoder.attach(tv)
    feature_encoder.encoders[6].set_simulate(True)
    tf = feature_encoder.transform(None)

    feature_encoder.attach(ev)
    feature_encoder.encoders[6].set_simulate(False)
    ef = feature_encoder.transform(None).set_index("process_id")
    tf.to_csv(output_filepath / "tf.csv", index=False)
    ef.to_csv(output_filepath / "ef.csv", index=False)
    assert (ef.index == submission_format.index).all()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
