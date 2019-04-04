#!/usr/bin/env python
import sys
sys.path.append(".")
import logging
import click
import re
from functools import partial
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
from src.features.build_features import calc_target_mean_features,\
    calc_special2_features
from src.models import contest_models
from src.models.contest_models import \
    ContestKerasModel, ContestLgbModel, \
    ContestSegmentationLayer, ContestScalingLayer, ContestKFoldLayer, \
    ContestBlendedModel
from src.keras_utils import keras_initialize_random_state

def split_train_valid_test(train_test, min_validate_date):
    min_validate_date_1001 = min_validate_date - pd.to_timedelta("144 hours")
    train_idx = train_test[
        (train_test.is_train) 
        & (
            ((train_test.expected_phase_summary!='1001') & (train_test.timestamp_min < min_validate_date))
            |
            ((train_test.expected_phase_summary=='1001') & (train_test.timestamp_min < min_validate_date_1001))
        )
    ].index
    train_test = calc_target_mean_features(train_test, train_idx=train_idx)

    bad_columns = set(train_test.filter(regex='.*(last).*').columns.tolist())
    white_labelled = set(train_test.filter(regex='.*(most_likely).*').columns.tolist())
    bad_columns = bad_columns.difference(white_labelled)
    logging.info(f"removing columns {bad_columns}")
    train_test = train_test[[c for c in train_test.columns if c not in bad_columns]]

    train = train_test.loc[train_idx]
    valid = train_test[(train_test.is_train) & ~(train_test.index.isin(train_idx))]
    test = train_test[~train_test.is_train]
    return train, valid, test


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True), default="data/processed/train_test.hdf5")
@click.option("--gen-submission", is_flag=True)
@click.option("--submission-file", type=click.Path())
@click.option("--blend-with-submission", type=click.Path())
@click.option("--min-validate-date", default="2018-04-20")
@click.option("--random-state", type=int, default=0)
@click.option("--lib", type=click.Choice(['lgb', 'keras']), default='lgb')
@click.option("--model")
@click.option("--scaling/--no-scaling", is_flag=True)
@click.option("--clip-scaling/--no-clip-scaling", is_flag=True)
@click.option("--sel-segment")
@click.option("--sel-object-id", type=int)
@click.option("--epochs", type=int, default=100)
@click.option("--learning-rate", type=float, default=0.05)
@click.option("--early-stopping", type=int, default=20)
@click.option("--patience", type=int, default=50)
@click.option("--num-boost-round", type=int, default=100)
@click.option("--min-data-in-leaf", type=int, default=30)
@click.option("--k-fold", is_flag=True)
@click.option("--validation-plot/--no-validation-plot", is_flag=True, default=False)
@click.option("--remove-outliers/--no-remove-outliers", is_flag=True, default=True)
def main(input_filepath, gen_submission, submission_file, blend_with_submission, 
    min_validate_date, random_state, 
    lib, model, scaling, clip_scaling, sel_segment, sel_object_id, epochs, learning_rate,
    early_stopping, patience, num_boost_round, min_data_in_leaf,
    k_fold,
    validation_plot, remove_outliers):
    np.random.seed(0)
    logging.info(f"experiment sel_segment={sel_segment} sel_object_id={sel_object_id} min_validate_date={min_validate_date} random_state={random_state}")

    min_validate_date = pd.to_datetime(min_validate_date) if min_validate_date not in ["", "None"] else None
    train_test = pd.read_hdf(input_filepath, "train_test")
    train_test_events = pd.read_hdf("data/interim/train_test.hdf5", "train_test_events")

    # drop invalid columns!
    train_test = train_test.drop(train_test.filter(regex='.*_last.*').columns, axis=1)

    #f_phase_4_clean_water_acid
    #f_phase_3_clean_water_recovery_water

    logging.info("train_test.size=%d", len(train_test))

    if remove_outliers:
        logging.info("removing outliers")

        # object_id=102, first process, very big target value
        train_test = train_test[~train_test.index.isin([26693])]

        train_test = train_test[
            ~(train_test.is_train)
            | (
                (train_test['f_phase_1_size'] < 4000)
                & (train_test['f_phase_2_size'] < 4000)
                & (train_test['f_phase_3_size'] < 4000)
                & (train_test['f_phase_4_size'] < 4000)
            )
        ]

        train_test = train_test[
            ~(train_test.is_train)
            | (train_test['f_phase_4_size'] == 0)
            | (train_test['f_phase_4_size'] >= 90)
        ]

        train_test = train_test[
            ~(train_test.is_train)
            | (train_test['pipeline'] != 12)
        ]

        train_test = train_test[
            ~(train_test.is_train)
            | (train_test.min_phase_num <= 4)
        ]


    logging.info("(after) train_test.size=%d", len(train_test))

    if sel_segment:
        train_test = train_test[train_test.expected_phase_summary.str.startswith(sel_segment)]

    if sel_object_id is not None:
        train_test = train_test[train_test.object_id==sel_object_id]

    # train_test = calc_special2_features(train_test, train_test_events)

    logging.info("splitting data to train/valid/test")
    if min_validate_date is not None:
        train, valid, test = split_train_valid_test(train_test, min_validate_date)
    else:
        train_test = calc_target_mean_features(train_test)
        train, valid, test = train_test[train_test.is_train], None, train_test[~train_test.is_train]

    # train_test = calc_knn_features_new(train_test)

    logging.info(f"train={len(train)}, valid={len(valid) if valid is not None else None} test={len(test)}")
    logging.info(f"train.columns={train.columns.tolist()}")

    features = list(sorted(train.filter(regex='^f_').columns.tolist()))
    logging.info(f"features_count={len(features)}")

    if lib == 'keras':
        model_cls = ContestKerasModel
        model_params = {
            'random_state': random_state,
            'epochs': epochs,
            'patience': patience
        }
    else:
        model_cls = ContestLgbModel
        model_params = {
            'random_state': random_state,
            'num_leaves': 32,
            'learning_rate': learning_rate,
            'early_stopping': early_stopping,
            'num_boost_round': num_boost_round,
            'min_data_in_leaf': min_data_in_leaf
        }
    if model is not None:
        model_cls = getattr(contest_models, model)

    if k_fold:
        model_params = {
            'model_factory': model_cls,
            'model_kwargs': model_params.copy(),
        }
        model_cls = ContestKFoldLayer

    if scaling:
        logging.info("using scaling")
        model_params = {
            'model_factory': model_cls, 
            'model_kwargs': model_params.copy()
        }
        model_cls = partial(ContestScalingLayer, limit_targets=clip_scaling)

    m = ContestSegmentationLayer(model_cls, model_kwargs=model_params)

    validation_score = m.train(train, valid, test)

    logging.info(f"validation score={validation_score:.4f}")
    logging.info(f"validation by segment:")
    logging.info(m.cv)

    now = pd.Timestamp.now()
    now_str = now.strftime('%Y%m%d-%H%M')


    if validation_plot:
        logging.info("generating validation plot")
        m.plot_validation_results(train, f"/tmp/results-{now_str}-train.png")
        if valid is not None:
            m.plot_validation_results(valid, f"/tmp/results-{now_str}-valid.png")

    if gen_submission:

        y_test_pred = m.predict(test)
        y_test_pred = y_test_pred.clip(train['target'].min(), train['target'].max())

        fn = (submission_file.replace(".csv", "") if submission_file else f"/tmp/submission-{now.strftime('%Y%m%d-%H%M')}") \
            + "-hist.png"
        logging.info(f"writing histogram to file {fn}")
        fig, ax = plt.subplots(figsize=(10, 5))
        y_test_pred.hist(bins=30, ax=ax)
        plt.savefig(fn, dpi=150)

        if blend_with_submission:
            submission = pd.read_csv(blend_with_submission)
        else:
            submission = pd.read_csv("data/raw/submission_format.csv")
            submission['final_rinse_total_turbidity_liter'] = None
        submission = submission.join(
            y_test_pred.dropna().rename("new"), 
            how='left', on='process_id'
        )
        submission['final_rinse_total_turbidity_liter'] = submission['new'].fillna(submission['final_rinse_total_turbidity_liter'])
        del submission['new']
        fn = submission_file if submission_file else f"/tmp/submission-{now.strftime('%Y%m%d-%H%M')}.csv"
        logging.info(f"writing submission to {fn}")
        submission.to_csv(fn, index=False)


if __name__ == "__main__":
    keras_initialize_random_state()
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_fmt)
    main()