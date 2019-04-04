import pytest
import pandas as pd

INPUT_FN = "data/interim/train_test.hdf5"
INTERIM_DATA = None


def test_iterim_counts(data):
    assert len(data) == 7_867_980
    assert len(data[data.is_train]) == 5_987_820
    assert len(data[~data.is_train]) == 1_880_160


def test_phase_num(data):
    assert all(data.phase_num.isin([1, 2, 3, 4, 5, 6]))


def test_pipeline(data):
    assert all(data.pipeline.isin(range(1, 13)))
    assert all(data[~data.is_train].pipeline.isin(range(1, 12)))


def test_object_id(data):
    # print(data.groupby(['object_id', 'is_train'])['process_id'].nunique().unstack(level=1))
    assert all(data.object_id.isin(range(102, 978)))


def test_return_turbidity(data):
    # print(data['return_turbidity'].describe())
    assert data['return_turbidity'].min() >= -0.4
    assert data['return_turbidity'].max() <= 101.2


def test_timestamp_ranges(data):
    TRAIN_START = pd.to_datetime("2018-02-21 17:42:18")
    TRAIN_END = pd.to_datetime("2018-04-25 12:02:12")
    TEST_START = pd.to_datetime("2018-04-25 10:57:58")
    TEST_END = pd.to_datetime("2018-05-24 14:26:57")

    assert all(data[data.is_train]['timestamp'].between(TRAIN_START, TRAIN_END))
    assert all(
        data[data.is_train].groupby('process_id')['timestamp'].min() < TEST_START
    )
    assert all(data[~data.is_train]['timestamp'].between(TEST_START, TEST_END))


def test_timestamp_diff(data):
    """time series are given in 2 seconds time intervals"""
    assert all(data.groupby('process_id')['timestamp'].diff().isin([pd.NaT, pd.to_timedelta("2 seconds")]))


def test_unique_timestamp_object_id(data):
    assert all(data.groupby(['timestamp', 'object_id']).size()==1)


def test_almost_unique_timestamp_pipeline(data):
    assert all(data.groupby(['timestamp', 'pipeline']).size().isin([1, 2]))


def test_supply_flags(data):
    g = data[['supply_pre_rinse', 'supply_caustic', 'supply_acid', 'supply_clean_water']].sum(axis=1)
    assert all(g.isin([0, 1, 2]))
    # process_id=23218 object_id=955 phase=acid => supply_acid=True, supply_clean_water=True
    assert len(g[g==2])==1


def test_return_flags(data):
    g = data[['return_caustic', 'return_acid', 'return_recovery_water', 'return_drain']].sum(axis=1)
    assert all(g.isin([0, 1]))


def test_objects(data):
    # all objects use the same pipeline
    assert all(data.groupby('object_id')['pipeline'].nunique()==1)


@pytest.fixture
def data(request):
    global INTERIM_DATA
    if INTERIM_DATA is None:
        print(f"reading data from {INPUT_FN}")
        INTERIM_DATA = pd.read_hdf(INPUT_FN, "train_test_events")
    return INTERIM_DATA