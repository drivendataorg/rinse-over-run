import re
import logging
from functools import partial
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.common import compress_bool_seq, weighted_median

def bool_columns():
    return [
        'supply_pump', 'supply_pre_rinse', 'supply_caustic', 
        'return_caustic', 'supply_acid', 'return_acid', 'supply_clean_water', 
        'return_recovery_water', 'return_drain', 'object_low_level', 'tank_lsh_caustic', 
        'tank_lsh_clean_water'
    ]

def temp_columns():
    return [
        'return_temperature',
        'tank_temperature_pre_rinse',
        'tank_temperature_caustic',
        'tank_temperature_acid'
    ]

def tank_level_columns():
    return [
        'tank_level_pre_rinse',
        'tank_level_caustic',
        'tank_level_acid',
        'tank_level_clean_water'
    ]


def calc_number_of_skipped_phases(df):
    p1 = df['expected_phase_summary'].str.extract("^(.)(.)(.)(.)$", expand=True) == '1' 
    p2 = df['phase_summary'].str.extract("^(.)(.)(.)(.)$", expand=True) != '0'
    skipped = (p1 != p2).applymap(int).cumsum(axis=1)

    for p in range(1, 5):
        col = f'f_phase_{p}_number_of_skipped_phases'
        df[col] = skipped[p - 1]
    return df


def calc_basic_features(df, train_labels, recipe_metadata):
    g = df.groupby('process_id')
    gp = df.groupby(['process_id', 'phase_num'])

    # TODO zmienić kolejność na {col}_{label}
    fp = lambda col, label, func: gp[col].apply(func).unstack(level=1).reindex(columns=range(1, 7)).rename(columns=lambda x: f"f_phase_{x}_{label}_{col}") 
    res = pd.concat([
        g[['object_id', 'is_train', 'pipeline']].first(),
        g['timestamp'].min().rename("timestamp_min"),
        g['timestamp'].max().rename("timestamp_max"),
        g['phase_num'].min().rename("min_phase_num"),
        g['phase_num'].max().rename("max_phase_num"),
        fp('timestamp', 'size', len).rename(columns=lambda x: x.replace('_timestamp', ''))\
            .fillna(0).astype('int'),
        fp('total_turbidity_liter', 'sum', np.sum).fillna(0).astype('float64')
    ], axis=1)

    res['f_object_id'] = (res['object_id']/100).apply(np.floor).astype('int8')


    for c in ['sum_total_turbidity_liter']:
        sel_columns = sorted(res.filter(regex='.*' + c + '$').columns.tolist())
        new_columns = list(map(lambda x: x.replace(c, 'cum_' + c), sel_columns))
        res = pd.concat(
            [res, res[sel_columns].cumsum(axis=1).rename(columns=dict(zip(sel_columns, new_columns)))],
            axis=1
        )

    res['target'] = train_labels.astype('float64')
    T = 290_000
    res['weight'] = 1 / res['target'].clip(T, None)
    assert all(res[res.is_train].target.notnull())

    res['phase_summary'] = (res[[f'f_phase_{i}_size' for i in range(1, 7)]] > 0)\
        .applymap(int).applymap(str)\
        .apply(lambda x: x.str.cat().rstrip('0')[:4].ljust(4, '?'), axis=1)

    res['expected_phase_summary'] = res.index.map(
        recipe_metadata[['pre_rinse', 'caustic', 'intermediate_rinse', 'acid']]\
        .applymap(str)\
        .apply(lambda x: x.str.cat(), axis=1)
    )

    p_stats = res[res.is_train].groupby(['expected_phase_summary', 'phase_summary'])\
        .size().rename("count").reset_index()

    curr = pd.concat([
        p_stats.groupby([p_stats.expected_phase_summary, p_stats.phase_summary.str[:i]]).apply(
           lambda x: x.sort_values(by='count', ascending=False).iloc[0].phase_summary
        )
        for i in range(0, 5)
    ], axis=0, sort=False)

    for i in range(1, 5):
        k = lambda x: (x.expected_phase_summary, x.phase_summary[:i].replace("?", ""))
        res[f'phase_{i}_most_likely_phase_summary'] = res[['expected_phase_summary', 'phase_summary']]\
            .apply(
                lambda x: curr.loc[k(x)] if k(x) in curr.index else x.expected_phase_summary,
                axis=1
            )

    res = calc_number_of_skipped_phases(res)

    return res


def calc_rle_features(df):
    logging.info("calculating rle_features")
    g = df[df.phase_num <= 4].groupby('process_id')
    res = []
    for b in tqdm(bool_columns()):
        res.append(
            g[b].apply(compress_bool_seq).rename(f"rle_{b}")
        )
    return pd.concat(res, axis=1)


def calc_agg_features(df):
    logging.info("normalizing tank_level columns")
    for c in tank_level_columns():
        logging.info(f"column {c}")
        g = df.groupby(['process_id', 'phase_num'])[c]
        min_values = g.transform(np.min)
        df[c] = ((df[c] - min_values) + (min_values >= 5.0) * 5.0) / 100.0

    logging.info("calculating conditional total_turbidity_liter")
    for c in bool_columns():
        df[f'total_turbidity_liter_{c}'] = df['total_turbidity_liter'].multiply(df[c])

    g = df.groupby('process_id')
    gp = df.groupby(['process_id', 'phase_num'])

    fp = lambda col, label, func: gp[col].apply(func).unstack(level=1).rename(columns=lambda x: f"f_phase_{x}_{label}_{col}") 

    rolling_diff_abs_sum = lambda x: x.map(float).rolling(window=20).mean().round(0)\
                            .diff().abs().sum()
    diff_abs_sum = lambda x: x.diff().abs().sum()
    change_func = lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 0 else None

    logging.info("calculating agg")
    res = []
    for column, label, func in tqdm([
        ('supply_flow', 'mean', np.mean),
        ('supply_flow', 'diff_abs_sum', diff_abs_sum),
        ('supply_pressure', 'mean', np.mean),
        ('supply_pressure', 'max', np.max),
        ('supply_pressure', 'diff_abs_sum', diff_abs_sum),
        ('tank_level_clean_water', 'change', change_func),
        ('tank_level_pre_rinse', 'change', change_func),
        ('tank_level_caustic', 'change', change_func),
        ('tank_level_acid', 'change', change_func),
        ('tank_level_clean_water', 'diff_abs_sum', diff_abs_sum),
        ('tank_level_pre_rinse', 'diff_abs_sum', diff_abs_sum),
        ('tank_level_clean_water', 'mean', np.mean),
        ('tank_level_pre_rinse', 'mean', np.mean),
        ('tank_level_clean_water', 'max', np.max),
        ('tank_level_pre_rinse', 'max', np.max),
        ('tank_level_clean_water', 'ptp', np.ptp),
        ('tank_level_pre_rinse', 'ptp', np.ptp),
        ('tank_concentration_caustic', 'mean', np.mean),
        ('tank_concentration_acid', 'mean', np.mean),
        ('return_temperature', 'mean', np.mean),
        ('return_temperature', 'max', np.max),
        ('return_temperature', 'diff_abs_sum', diff_abs_sum),
        ('return_conductivity', 'mean', np.mean),
        ('return_conductivity', 'max', np.max),
        ('return_flow', 'sum', np.sum),
        ('return_flow', 'mean', np.mean),
        ('return_flow', 'max', np.max),
        ('return_turbidity', 'sum', np.sum),
        ('return_turbidity', 'mean', np.mean),
        ('return_turbidity', 'max', np.max),
        ('total_turbidity_liter', 'mean', np.mean),
        ('total_turbidity_liter', 'ptp', np.ptp),
        ('total_turbidity_liter', 'std', np.std),
        ('total_turbidity_liter', 'diff_abs_sum', diff_abs_sum),
    ]):
        res.append(fp(column, label, func))

    res += [
        gp['total_turbidity_liter'].last().unstack(level=1).rename(columns=lambda x: f"f_phase_{x}_last_turbidity_liter"),
        gp['timestamp'].apply(np.ptp).unstack(level=1).rename(columns=lambda x: f"f_phase_{x}_time"),
    ]

    logging.info("adding sums based on supply/return label")
    res.append(
        pd.concat([
            df[(df.phase_num==p) & (df.return_label==rl) & (df.supply_label==sl)].groupby('process_id')\
            ['total_turbidity_liter'].sum().rename(f"f_phase_{p}_sum_total_turbidity_liter_rl_{rl}_sl_{sl}")
            for p in range(1, 5) 
            for rl in df['return_label'].unique()
            for sl in df['supply_label'].unique()
        ], axis=1).fillna(0)
    )

    logging.info("adding bool columns")
    for c in tqdm(bool_columns()):
        res.append(fp(c, 'mean', np.mean))
        res.append(fp(c, 'sum', np.sum))
        res.append(fp(c, 'rolling_diff_abs_sum', rolling_diff_abs_sum))
        res.append(fp(f'total_turbidity_liter_{c}', 'sum', np.sum))

    res = pd.concat(res, axis=1)

    if False:
        tmp = res.groupby('object_id').size()
        top_objects = tmp[tmp >= 100].index
        res['top_object_id'] = res['object_id'].map(lambda x: x if x in top_objects else 0)

        res = pd.concat([
            res, 
            pd.get_dummies(res['pipeline'], prefix='f_pipeline'),
            pd.get_dummies(res['top_object_id'], prefix='f_top_object_id')
        ], axis=1)

    for c in res.filter(regex='.*_time$').columns:
        res[c] = res[c] / pd.to_timedelta("1 second")

    return res


def calc_time_features(df):
    df = df.sort_values(by=['object_id', 'timestamp_min'])
    df['f_days_since_prev_cleaning'] = df.groupby('object_id')['timestamp_min'].diff()
    df['f_days_since_prev_cleaning'] = df['f_days_since_prev_cleaning'] / pd.Timedelta(days=1)
    df['f_days_since_first_cleaning'] = (df['timestamp_min'] - df.groupby('object_id')['timestamp_min'].transform(np.min)) / pd.Timedelta(days=1)
    df['f_is_first_cleaning'] = df['f_days_since_prev_cleaning'].isnull()
    df['f_total_cleaning_num'] = df.groupby('object_id').cumcount()
    df['f_daily_cleaning_num'] = df.groupby([df['object_id'], df['timestamp_min'].dt.date]).cumcount()
    df['f_dayofweek_min'] = df['timestamp_min'].dt.dayofweek
    df['f_hour_min'] = df['timestamp_min'].dt.hour
    return df


def calc_target_mean_features_old(df, train_idx=None):
    if train_idx is not None:
        df['target_copy'] = None
        df.loc[train_idx, 'target_copy'] = df.loc[train_idx, 'target']
    else:
        df['target_copy'] = df['target']

    m1 = df['target_copy'].median()
    m2 = df['target'].median()
    global_mean_series = df['is_train'].map({True: m1, False: m2})

    df = df.sort_values(by='timestamp_min')

    def calc(g, f, shift=1):
        part1 = df[df.is_train].groupby(g)['target_copy'].apply(lambda x: f(x.shift(1)))
        part2 = df.groupby(g)['target'].apply(lambda x: f(x.shift(shift)))
        res = pd.concat([
            part1.loc[df[df.is_train].index],
            part2.loc[df[~df.is_train].index],
        ], axis=0)
        return res

    for c in ['object_id', 'pipeline']:
        cc = "_".join(c) if isinstance(c, list) else c
        df[f'f_{cc}_target_mean'] = calc(c, lambda x: x.expanding().mean())
        df[f'f_{cc}_target_last'] = calc(c, lambda x: x.fillna(method='ffill'))
        g0 = [df.object_id]
        g = [df.object_id, df.expected_phase_summary]
        df[f'f_object_id_exp_phase_target_mean'] = calc(g, lambda x: x.expanding().mean())
        df[f'f_object_id_exp_phase_target_last'] = calc(g, lambda x: x.fillna(method='ffill'))
        for p in range(1, 5):
            gp = [df['object_id'], df[f'phase_{p}_most_likely_phase_summary']]
            df[f'f_phase_{p}_object_id_most_likely_target_mean'] = \
                calc(gp, lambda x: x.expanding().median())\
                .fillna(calc(g0, lambda x: x.expanding().median()))\
                .fillna(global_mean_series)

    for p in range(1, 5):
        g0 = [df.object_id]
        g1 = [df.object_id, df.phase_summary.str[:p]]
        g2 = [df.pipeline, df.phase_summary.str[:p]]
        f_mean = lambda x: x.expanding().median()
        df[f'f_phase_{p}_pipeline_target_mean'] = calc(g2, f_mean)\
            .fillna(global_mean_series)
        df[f'f_phase_{p}_object_id_target_mean'] = calc(g1, f_mean)\
            .fillna(calc(g2, f_mean))\
            .fillna(calc(g0, f_mean))\
            .fillna(global_mean_series)

        if False:  # those values are unreliable on test set and too good on training
            f_last = lambda x: x.fillna(method='ffill')
            df[f'f_phase_{p}_object_id_target_last'] = calc(g1, f_last)
            df[f'f_phase_{p}_object_id_target_last2'] = calc(g1, f_last, shift=2)
            df[f'f_phase_{p}_object_id_target_last_diff'] = \
                df[f'f_phase_{p}_object_id_target_last'] - \
                df[f'f_phase_{p}_object_id_target_last2']

    if False:
        n_components = 2
        sel_columns = df.filter(regex='f_phase_[1-4]_(sum_total_turbidity_liter)').columns.tolist()
        sel = df[df.target_copy.notnull()]
        p = PCA(n_components=n_components).fit(sel[sel_columns], sel['target'])
        arr = p.transform(df[sel_columns])
        for i in range(n_components):
            df[f'f_pca_{i}'] = arr[:, i]

    del df['target_copy']
    
    return df


def calc_target_mean_features(df, train_idx=None, max_phase_num=4):
    df = df[[c for c in df.columns if not re.match('.*(_target|last).*', c)]].copy()
    train = df[df.is_train].sort_values(by='timestamp_min')
    test = df[~df.is_train]
    n_splits = 10
    parts = np.array_split(train.index, n_splits)

    def calc_feat(sel, train, q=0.5):
        print("calc for {}-{} using {}-{}".format(
            sel['timestamp_min'].min(), sel['timestamp_min'].max(), 
            train['timestamp_min'].min(), train['timestamp_min'].max(), 

        ))
        m1 = train.groupby(['object_id', 'expected_phase_summary'])\
            .apply(lambda x: weighted_median(x['target'].values, x['weight'].values, q))
        m2 = train.groupby(['pipeline', 'expected_phase_summary'])\
            .apply(lambda x: weighted_median(x['target'].values, x['weight'].values, q))
        global_median = weighted_median(train['target'].values, train['weight'].values, q)

        tmp = sel.join(
            m1.rename("m1"), on=['object_id', 'expected_phase_summary'], how='left'
        ).join(
            m2.rename("m2"), on=['pipeline', 'expected_phase_summary'], how='left'
        )
        return tmp['m1'].fillna(tmp['m2']).fillna(global_median)

    for q in [0.01, 0.5, 0.99]:
        res = []
        for part in parts:
            sel_part = train[train.index.isin(part)]
            other_part = train[~train.index.isin(part)]
            res.append(calc_feat(sel_part, other_part, q))
        res.append(calc_feat(test, train, q))
        res = pd.concat(res, axis=0)
        df[f'f_phase_1_target_median_{q:.2f}'] = res
    return df




def calc_target_mean_features_old2(df, train_idx=None, max_phase_num=4):
    org_idx = df.index

    if train_idx is not None:
        df['target_copy'] = None
        df.loc[train_idx, 'target_copy'] = df.loc[train_idx, 'target']
    else:
        df['target_copy'] = df['target']

    m1 = df['target_copy'].median()
    m2 = df['target'].median()
    global_mean_series = df['is_train'].map({True: m1, False: m2})

    df['sort_key'] = (~df['is_train']) * 2 + (~df.index.isin(train_idx if train_idx is not None else [])) * 1

    df = df.sort_values(by=['sort_key', 'timestamp_min'])

    def calc(g, f, shift=1):
        part1 = df[df.is_train].groupby(g)['target_copy'].apply(lambda x: f(x.shift(1)))
        part2 = df.groupby(g)['target'].apply(lambda x: f(x.shift(shift)))
        res = pd.concat([
            part1.loc[df[df.is_train].index],
            part2.loc[df[~df.is_train].index],
        ], axis=0)
        return res

    for c in ['object_id', 'pipeline']:
        cc = "_".join(c) if isinstance(c, list) else c
        df[f'f_{cc}_target_mean'] = calc(c, lambda x: x.expanding().mean())
        g0 = [df.object_id]
        g = [df.object_id, df.expected_phase_summary]
        df[f'f_object_id_exp_phase_target_mean'] = calc(g, lambda x: x.expanding().mean())
        for p in range(1, max_phase_num + 1):
            gp = [df['object_id'], df[f'phase_{p}_most_likely_phase_summary']]
            df[f'f_phase_{p}_object_id_most_likely_target_mean'] = \
                calc(gp, lambda x: x.expanding().median())
                #.fillna(calc(g0, lambda x: x.expanding().median()))\
                #.fillna(global_mean_series)

    for p in range(1, max_phase_num + 1):
        g0 = [df.object_id]
        g1 = [df.object_id, df.phase_summary.str[:p]]
        g2 = [df.pipeline, df.phase_summary.str[:p]]
        f_mean = lambda x: x.expanding().median()
        df[f'f_phase_{p}_pipeline_target_mean'] = calc(g2, f_mean)
            #.fillna(global_mean_series)
        df[f'f_phase_{p}_object_id_target_mean'] = calc(g1, f_mean)
            #.fillna(calc(g2, f_mean))\
            #.fillna(calc(g0, f_mean))\
            #.fillna(global_mean_series)

    del df['target_copy']
    del df['sort_key']
    
    return df.reindex(org_idx)


def calc_special_features(train_test, train_test_events):
    train_test['f_phase_4_special_1'] = train_test['f_phase_1_sum_return_flow'] + train_test['f_phase_4_sum_total_turbidity_liter_tank_lsh_caustic']
    train_test['f_phase_4_special_2'] = train_test['f_phase_4_sum_return_flow'] + train_test['f_phase_4_sum_total_turbidity_liter_tank_lsh_caustic']
    train_test['f_phase_3_special_3'] = (train_test['f_phase_3_sum_total_turbidity_liter'] - train_test['f_phase_3_sum_total_turbidity_liter_return_caustic']).abs()

    f = lambda df, phase_num, supply_label, return_label: \
        df[(df.phase_num==phase_num) & (df.supply_label==supply_label) & (df.return_label==return_label)]\
        .groupby('process_id')['total_turbidity_liter'].sum()\
        .fillna(0)\
        .rename(f"f_phase_total_turbidity_liter_{phase_num}_{supply_label}_{return_label}")\
        .reindex(train_test.index)

    train_test['f_phase_4_special_4'] = f(train_test_events, 4, "acid", "acid")
    train_test['f_phase_4_special_5'] = f(train_test_events, 4, "none", "acid")
    train_test['f_phase_4_special_6'] = f(train_test_events, 4, "clean_water", "acid")
    train_test['f_phase_3_special_7'] = f(train_test_events, 3, "clean_water", "recovery_water")
    return train_test

def calc_special2_features(train_test, train_test_events):
    g = train_test_events.groupby('process_id')
    gp = train_test_events.groupby(['process_id', 'phase_num'])

    fp = lambda col, label, func: gp[col].apply(func).unstack(level=1).rename(columns=lambda x: f"f_phase_{x}_{label}_{col}") 

    res = []
    logging.info("adding rle like features")
    for rle in ['01', '0', '1']:
        for c in ['rle_supply_pre_rinse']:
            res.append(
                (train_test[c] == rle).astype(int).rename(f'f_phase_1_{c}_{rle}')
            )
    logging.info("adding total_turbidity_liter median feat")
    for column, label, func in tqdm([
        ('total_turbidity_liter', 'neg_sum', lambda x: x.clip(None, 0).abs().sum()),
        ('total_turbidity_liter', 'median', np.median),
        ('total_turbidity_liter', 'q_0_1', partial(np.quantile, q=0.1)),
        ('total_turbidity_liter', 'q_0_9', partial(np.quantile, q=0.9)),
    ]):
        res.append(fp(column, label, func))
    res = pd.concat(res, axis=1)

    train_test = train_test.join(res, how='left')

    return train_test


def calc_target_mean_features_as_of_20190222(df, train_idx=None):
    if train_idx is not None:
        df['target_copy'] = None
        df.loc[train_idx, 'target_copy'] = df.loc[train_idx, 'target']
    else:
        df['target_copy'] = df['target']

    m1 = df['target_copy'].mean()
    m2 = df['target'].mean()
    global_mean_series = df['is_train'].map({True: m1, False: m2})

    df = df.sort_values(by='timestamp_min')

    def calc(g, f, shift=1):
        part1 = df[df.is_train].groupby(g)['target_copy'].apply(lambda x: f(x.shift(1)))
        part2 = df.groupby(g)['target'].apply(lambda x: f(x.shift(shift)))
        res = pd.concat([
            part1.loc[df[df.is_train].index],
            part2.loc[df[~df.is_train].index],
        ], axis=0)
        return res

    for c in ['object_id', 'pipeline']:
        cc = "_".join(c) if isinstance(c, list) else c
        df[f'f_{cc}_target_mean'] = calc(c, lambda x: x.expanding().mean())
        df[f'f_{cc}_target_last'] = calc(c, lambda x: x.fillna(method='ffill'))
        g0 = [df.object_id]
        g = [df.object_id, df.expected_phase_summary]
        df[f'f_object_id_exp_phase_target_mean'] = calc(g, lambda x: x.expanding().mean())
        df[f'f_object_id_exp_phase_target_last'] = calc(g, lambda x: x.fillna(method='ffill'))
        for p in range(1, 5):
            gp = [df['object_id'], df[f'phase_{p}_most_likely_phase_summary']]
            df[f'f_phase_{p}_object_id_most_likely_target_mean'] = \
                calc(gp, lambda x: x.expanding().mean())
                #.fillna(calc(g0, lambda x: x.expanding().mean()))\
                #.fillna(global_mean_series)

    for p in range(1, 5):
        g0 = [df.object_id]
        g1 = [df.object_id, df.phase_summary.str[:p]]
        g2 = [df.pipeline, df.phase_summary.str[:p]]
        f_mean = lambda x: x.expanding().mean()
        f_last = lambda x: x.fillna(method='ffill')
        df[f'f_phase_{p}_pipeline_target_mean'] = calc(g2, f_mean)
            #.fillna(global_mean_series)
        df[f'f_phase_{p}_object_id_target_mean'] = calc(g1, f_mean)\
            .fillna(calc(g2, f_mean))
            #.fillna(calc(g0, f_mean))\
            #.fillna(global_mean_series)
        df[f'f_phase_{p}_object_id_target_last'] = calc(g1, f_last)
        df[f'f_phase_{p}_object_id_target_last2'] = calc(g1, f_last, shift=2)
        df[f'f_phase_{p}_object_id_target_last_diff'] = \
            df[f'f_phase_{p}_object_id_target_last'] - \
            df[f'f_phase_{p}_object_id_target_last2']

    if False:
        n_components = 2
        sel_columns = df.filter(regex='f_phase_[1-4]_(sum_total_turbidity_liter)').columns.tolist()
        sel = df[df.target_copy.notnull()]
        p = PCA(n_components=n_components).fit(sel[sel_columns], sel['target'])
        arr = p.transform(df[sel_columns])
        for i in range(n_components):
            df[f'f_pca_{i}'] = arr[:, i]

    del df['target_copy']
    
    return df


def calc_features(df, train_labels, recipe_metadata):
    res = pd.concat([
        calc_basic_features(df, train_labels, recipe_metadata),
        # calc_rle_features(df),
        calc_agg_features(df)
    ], axis=1)
    res = calc_time_features(res)
    # TW 2019-03-20 changed to use very old version of this code
    res = calc_target_mean_features_as_of_20190222(res)
    # res = calc_target_mean_features(res)
    res = calc_special_features(res, df)
    return res

