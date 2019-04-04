import pandas as pd
import numpy as np
from tqdm import tqdm

## Load data
phase_list = ['pre_rinse', 'caustic', 'intermediate_rinse', 'acid']
train = pd.read_csv('data/train_values.csv', index_col='row_id', parse_dates=['timestamp'])
test = pd.read_csv('data/test_values.csv', index_col='row_id', parse_dates=['timestamp'])
train_labels = pd.read_csv('data/train_labels.csv', index_col='process_id')
recipes = pd.read_csv('data/recipe_metadata.csv', index_col='process_id').drop(['final_rinse'], axis=1)

# Drop columns with no variation
train.drop(['tank_lsh_acid', 'tank_lsh_pre_rinse', 'tank_lsh_clean_water'], axis=1, inplace=True)
test.drop(['tank_lsh_acid', 'tank_lsh_pre_rinse', 'tank_lsh_clean_water'], axis=1, inplace=True)

# Generate datasets for each phase, assign new process IDs
new_train = []
for i,p in enumerate(tqdm(phase_list)):    
    df = train[(train['phase'].isin(phase_list[:i+1]))].copy()
    df['max_phase'] = p
    df['last_phase'] = df['process_id'].map(df.groupby('process_id')['phase'].last())
    df['orig_process_id'] = df['process_id'].copy()
    df['process_id'] = df['max_phase']+'-'+df['process_id'].astype('str')
    new_train.append(df)
train = pd.concat(new_train)
test['last_phase'] = test['process_id'].map(test.groupby('process_id')['phase'].last())

# Add turbidity
train['turbidity'] = np.maximum(train['return_flow'], 0) * train['return_turbidity']
test['turbidity'] = np.maximum(test['return_flow'], 0) * test['return_turbidity']


# Get dataframe of process IDs
train_df = train[['process_id', 'max_phase', 'last_phase', 'orig_process_id']].drop_duplicates().set_index('process_id')
test_df = test[['process_id', 'last_phase']].drop_duplicates().set_index('process_id')

## Add duration stats
def proc_duration(df):
    df = df.pivot_table(values='timestamp', index='process_id', columns='phase', aggfunc='count')
    df = df[[c for c in phase_list if c in df.columns]]
    df[df.isnull()]=0
    for c in df.columns:
        df[c+'_pct'] = df[c]/df.sum(axis=1)
    return df

train_df = pd.concat([train_df, proc_duration(train)], axis=1, sort=False)
test_df = pd.concat([test_df, proc_duration(test)], axis=1, sort=False)

## Process categorical features
cat_cols = ['pipeline', 'object_id']
for c in cat_cols:
    train_df[c] = train[['process_id', c]].drop_duplicates().set_index('process_id')[c]
    train_df.loc[:,c] = train_df.loc[:,c].astype('category').cat.as_ordered()
    
    test_df[c] = test[['process_id', c]].drop_duplicates().set_index('process_id')[c]
    test_df.loc[:,c] = pd.Categorical(test_df[c], categories=train_df[c].cat.categories, ordered=True)

## Add recipes as categorical
recipes['recipe'] = ['-'.join([c[0] if recipes.loc[i, c] > 0 else '_' for c in phase_list]) for i in recipes.index]
cat_cols += ['recipe']

train_df['recipe'] = train_df['orig_process_id'].map(recipes['recipe'])
test_df['recipe'] = test_df.index.map(recipes['recipe'])

train_df['recipe'] = train_df['recipe'].astype('category').cat.as_ordered()
test_df['recipe'] = pd.Categorical(test_df['recipe'], categories=train_df['recipe'].cat.categories, ordered=True)

## Add time series/boolean-derived features
def prep_time_series_features(df, ts_cols):
    def L5(x): return x.tail(5).mean()
    agg_list = ['min', 'max', 'mean', 'median', 'skew', 'std', L5]
    ts_features = df.set_index('process_id')[ts_cols].groupby(['process_id']).agg(agg_list)
    ts_features.columns = [str(x).replace('(', '').replace(')', '').replace("'", '').replace(", ", '_') for x in ts_features.columns]
    return ts_features

ts_cols = [c for c in train.columns.tolist() if c not in ['process_id', 'object_id', 'phase', 'timestamp', 'pipeline', 'target_time_period', 'orig_process_id', 'max_phase', 'last_phase', 'target']]
train_df = pd.concat([train_df, prep_time_series_features(train, ts_cols=ts_cols)], axis=1, sort=False)
test_df = pd.concat([test_df, prep_time_series_features(test, ts_cols=ts_cols)], axis=1, sort=False)

## Save
train_df['target'] = train_df['orig_process_id'].map(train_labels['final_rinse_total_turbidity_liter'])
train_df.reset_index().to_feather('tmp/train.feather')
test_df.reset_index().to_feather('tmp/test.feather')