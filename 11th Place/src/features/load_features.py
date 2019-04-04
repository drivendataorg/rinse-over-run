import cPickle as pickle
import pandas as pd


def load_train_test(DATA_DIR, **params):
    train = []
    test = []
    for key, value in params.iteritems():
        print "%s = %s" % (key, value)

    if params["categorical"]:
        with open(DATA_DIR + 'pickle/train_categorical.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_categorical.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load timeseries last n values
    for i in params["timeseries"]:
        with open(DATA_DIR + 'pickle/train_ts_%d.p' % i, 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_ts_%d.p' % i, 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load all time series
    if params["ts_all"]:
        with open(DATA_DIR + 'pickle/train_ts_all.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_ts_all.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load time data
    if params["time"]:
        with open(DATA_DIR + 'pickle/train_time.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_time.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["pipeline_time"]:
        with open(DATA_DIR + 'pickle/train_pipeline_time.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_pipeline_time.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["phase_data"]:
        with open(DATA_DIR + 'pickle/train_phase.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_phase.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["object_data"]:
        with open(DATA_DIR + 'pickle/train_object.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_object.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["boolean_data"]:
        with open(DATA_DIR + 'pickle/train_boolean.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_boolean.p', 'rb') as fdf:
            test.append(pickle.load(fdf))
    if params["operating"]:
        with open(DATA_DIR + 'pickle/train_operating.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_operating.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    train_features = pd.concat(train, axis=1)
    test_features = pd.concat(test, axis=1)

    return train_features, test_features


def load_train_min_test(DATA_DIR, **params):
    train = []
    test = []
    for key, value in params.iteritems():
        print "%s = %s" % (key, value)

    if params["categorical"]:
        with open(DATA_DIR + 'pickle/train_min_categorical.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_min_categorical.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load timeseries last n values
    for i in params["timeseries"]:
        with open(DATA_DIR + 'pickle/train_min_ts_%d.p' % i, 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_min_ts_%d.p' % i, 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load all time series
    if params["ts_all"]:
        with open(DATA_DIR + 'pickle/train_min_ts_all.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_min_ts_all.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load time data
    if params["time"]:
        with open(DATA_DIR + 'pickle/train_min_time.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_min_time.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["pipeline_time"]:
        with open(DATA_DIR + 'pickle/train_min_pipeline_time.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_min_pipeline_time.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["phase_data"]:
        with open(DATA_DIR + 'pickle/train_min_phase.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_min_phase.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["object_data"]:
        with open(DATA_DIR + 'pickle/train_min_object.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_min_object.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["boolean_data"]:
        with open(DATA_DIR + 'pickle/train_min_boolean.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_min_boolean.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    train_features = pd.concat(train, axis=1)
    test_features = pd.concat(test, axis=1)

    return train_features, test_features


def load_train_outlier_test(DATA_DIR, **params):
    train = []
    test = []
    for key, value in params.iteritems():
        print "%s = %s" % (key, value)

    if params["categorical"]:
        with open(DATA_DIR + 'pickle/train_outlier_categorical.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_categorical.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load timeseries last n values
    for i in params["timeseries"]:
        with open(DATA_DIR + 'pickle/train_outlier_ts_%d.p' % i, 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_ts_%d.p' % i, 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load all time series
    if params["ts_all"]:
        with open(DATA_DIR + 'pickle/train_outlier_ts_all.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_ts_all.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load time data
    if params["time"]:
        with open(DATA_DIR + 'pickle/train_outlier_time.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_time.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["pipeline_time"]:
        with open(DATA_DIR + 'pickle/train_outlier_pipeline_time.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_pipeline_time.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["phase_data"]:
        with open(DATA_DIR + 'pickle/train_outlier_phase.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_phase.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["object_data"]:
        with open(DATA_DIR + 'pickle/train_outlier_object.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_object.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["boolean_data"]:
        with open(DATA_DIR + 'pickle/train_outlier_boolean.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_boolean.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["operating"]:
        with open(DATA_DIR + 'pickle/train_outlier_operating.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_operating.p', 'rb') as fdf:
            test.append(pickle.load(fdf))
    train_features = pd.concat(train, axis=1)
    test_features = pd.concat(test, axis=1)

    return train_features, test_features


def load_train_outlier_plus_test(DATA_DIR, **params):
    train = []
    test = []
    for key, value in params.iteritems():
        print "%s = %s" % (key, value)

    if params["categorical"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_categorical.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_categorical.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load timeseries last n values
    for i in params["timeseries"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_ts_%d.p' % i, 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_ts_%d.p' % i, 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load all time series
    if params["ts_all"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_ts_all.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_ts_all.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    # load time data
    if params["time"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_time.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_time.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["pipeline_time"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_pipeline_time.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_pipeline_time.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["phase_data"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_phase.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_phase.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["object_data"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_object.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_object.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["boolean_data"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_boolean.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_boolean.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    if params["operating"]:
        with open(DATA_DIR + 'pickle/train_outlier_2_operating.p', 'rb') as fdf:
            train.append(pickle.load(fdf))

        with open(DATA_DIR + 'pickle/test_outlier_2_operating.p', 'rb') as fdf:
            test.append(pickle.load(fdf))

    train_features = pd.concat(train, axis=1)
    test_features = pd.concat(test, axis=1)

    return train_features, test_features
