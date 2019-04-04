import pytest
import numpy as np
import pandas as pd

PR_INPUT_FN = "data/processed/train_test.hdf5"
PROCESSED_DATA = None

def test_processed_counts(data):
    assert len(data) == 7_988
    assert len(data[data.is_train]) == 5_021
    assert len(data[~data.is_train]) == 2_967


def test_f_total_cleaning_num(data):
    g = data.groupby('object_id')
    df = pd.concat([
        g.size().rename("count"),
        g['f_total_cleaning_num'].nunique().rename("nunique"),
        g['f_total_cleaning_num'].min().rename("min"),
        g['f_total_cleaning_num'].max().rename("max"),
    ], axis=1)
    assert all(df['count']==df['nunique'])
    assert all(df['min']==0)
    assert all(df['max']==df['count'] - 1)
    tmp = g.apply(lambda x: x.sort_values(by='timestamp_min')['f_total_cleaning_num'].diff())
    assert all(tmp.isin([1, np.nan]))


def test_target(data):
    assert 359 <= data['target'].min() <= 360
    assert 1.47e8 <= data['target'].max() <= 1.48e8
    assert all(data[data.is_train]['target'].notnull())
    assert all(data[~data.is_train]['target'].isnull())


def test_f_phase_size(data):
    sel = data.filter(regex=r'f_phase_\d_size').stack().rename("size").reset_index()
    assert all(sel['size'].notnull())
    assert all(sel['size'] >= 0)
    sel = sel[sel['size'] > 0]
    assert all(sel['size'].between(2, 10507))
    for num, min_value, median_value, max_value in [
        (1, 5, 86, 2972),
        (2, 6, 442, 3089),
        (3, 22, 87, 1259),
        (4, 2, 258, 10507),
    ]:
        s = data[f'f_phase_{num}_size']
        s = s[s > 0]
        assert all(s.between(min_value, max_value))
        assert not all(s.between(min_value+1, max_value-1))
        assert s.median() == median_value


@pytest.fixture
def data(request):
    global PROCESSED_DATA
    if PROCESSED_DATA is None:
        print(f"reading data from {PR_INPUT_FN}")
        PROCESSED_DATA = pd.read_hdf(PR_INPUT_FN, "train_test")
    return PROCESSED_DATA