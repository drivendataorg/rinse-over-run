from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
pd_cols = [
    'process_id',
    'phase',
    'supply_flow',
    'supply_pressure',
    'return_temperature',
    'conductivity',
    'return_turbidity',
    'return_flow',
    'tank_level_pre_rinse',
    'tank_level_caustic',
    'tank_level_acid',
    'tank_level_clean_water',
    'tank_temperature_pre_rinse',
    'tank_temperature_caustic',
    'tank_temperature_acid',
    'tank_concentration_caustic',
    'tank_concentration_acid',
    'return_flow__return_turbidity',
    'object_residue',
    'return_flow__return_temperature',
    'return_flow__return_conductivity',
    'supply_flow_*_supply_pressure',
    'supply_flow_*_return_conductivity',
    'supply_flow_*_return_turbidity',
    'supply_flow_*_return_temperature',
    'conductivity_*_return_temperature',
    'conductivity_*_return_turbidity',
    'return_temperature_*_return_turbidity'
]
data_cols = [
    'object_id',
    'supply_flow',
    'supply_pressure',
    'return_temperature',
    'return_conductivity',
    'return_turbidity',
    'return_flow',
    'tank_level_pre_rinse',
    'tank_level_caustic',
    'tank_level_acid',
    'tank_level_clean_water',
    'tank_temperature_pre_rinse',
    'tank_temperature_caustic',
    'tank_temperature_acid',
    'tank_concentration_caustic',
    'tank_concentration_acid',
]
ts_cols = [
    'process_id',
    'supply_flow',
    'supply_pressure',
    'return_temperature',
    'conductivity',
    'return_turbidity',
    'return_flow',
    'tank_level_pre_rinse',
    'tank_level_caustic',
    'tank_level_acid',
    'tank_level_clean_water',
    'tank_temperature_pre_rinse',
    'tank_temperature_caustic',
    'tank_temperature_acid',
    'tank_concentration_caustic',
    'tank_concentration_acid'
]

top50 = ['object_405', 'object_932', 'object_933', 'object_934', 'object_204', 'object_112', 'object_930', 'object_113', 'object_921', 'object_216', 'object_922', 'object_923', 'object_306', 'object_924', 'object_217', 'object_308', 'object_116', 'object_940', 'object_942', 'object_943', 'object_944', 'object_941', 'object_309', 'object_114', 'object_300',
         'object_301', 'object_955', 'object_925', 'object_303', 'object_956', 'object_302', 'object_115', 'object_945', 'object_958', 'object_102', 'object_215', 'object_431', 'object_205', 'object_959', 'object_963', 'object_912', 'object_964', 'object_962', 'object_914', 'object_911', 'object_305', 'object_917', 'object_961', 'object_913', 'object_212']


def polyfeatures(df, p):
  new_features = df[[('supply_flow', 'mean'),
                     ('supply_pressure', 'mean'),
                     ('return_temperature', 'mean'),
                     ('return_conductivity', 'mean'),
                     ('return_turbidity', 'mean'),
                     ('return_flow', 'mean'),
                     ('tank_level_pre_rinse', 'mean'),
                     ('tank_level_caustic', 'mean'),
                     ('tank_level_acid', 'mean'),
                     ('tank_level_clean_water', 'mean'),
                     ('tank_temperature_pre_rinse', 'mean'),
                     ('tank_temperature_caustic', 'mean'),
                     ('tank_temperature_acid', 'mean'),
                     # ('supply_flow', 'sem'),
                     # ('supply_pressure', 'sem'),
                     # ('return_temperature', 'sem'),
                     # ('return_conductivity', 'sem'),
                     # ('return_turbidity', 'sem'),
                     # ('return_flow', 'sem'),
                     # ('tank_level_pre_rinse', 'sem'),
                     # ('tank_level_caustic', 'sem'),
                     # ('tank_level_acid', 'sem'),
                     # ('tank_level_clean_water', 'sem'),
                     # ('tank_temperature_pre_rinse', 'sem'),
                     # ('tank_temperature_caustic', 'sem'),
                     # ('tank_temperature_acid', 'sem'),
                     ('supply_flow', 'std'),
                     ('supply_pressure', 'std'),
                     ('return_temperature', 'std'),
                     ('return_conductivity', 'std'),
                     ('return_turbidity', 'std'),
                     ('return_flow', 'std'),
                     ('tank_level_pre_rinse', 'std'),
                     ('tank_level_caustic', 'std'),
                     ('tank_level_acid', 'std'),
                     ('tank_level_clean_water', 'std'),
                     ('tank_temperature_pre_rinse', 'std'),
                     ('tank_temperature_caustic', 'std'),
                     ('tank_temperature_acid', 'std'),
                     ('supply_flow', 'max'),
                     ('supply_pressure', 'max'),
                     ('return_temperature', 'max'),
                     ('return_conductivity', 'max'),
                     ('return_turbidity', 'max'),
                     ('return_flow', 'max'),
                     ('tank_level_pre_rinse', 'max'),
                     ('tank_level_caustic', 'max'),
                     ('tank_level_acid', 'max'),
                     ('tank_level_clean_water', 'max'),
                     ('tank_temperature_pre_rinse', 'max'),
                     ('tank_temperature_caustic', 'max'),
                     ('tank_temperature_acid', 'max'),
                     ('supply_flow', 'min'),
                     ('supply_pressure', 'min'),
                     ('return_temperature', 'min'),
                     ('return_conductivity', 'min'),
                     ('return_turbidity', 'min'),
                     ('return_flow', 'min'),
                     ('tank_level_pre_rinse', 'min'),
                     ('tank_level_caustic', 'min'),
                     ('tank_level_acid', 'min'),
                     ('tank_level_clean_water', 'min'),
                     ('tank_temperature_pre_rinse', 'min'),
                     ('tank_temperature_caustic', 'min'),
                     ('tank_temperature_acid', 'min'),
                     ]]
  pol = PolynomialFeatures(p)
  new_featuresf = pol.fit_transform(new_features)
  new_feat = pd.DataFrame(new_featuresf,
                          columns=pol.get_feature_names(input_features=['supply_flow_mean',
                                                                        'supply_pressure_mean',
                                                                        'return_temperature_mean',
                                                                        'return_conductivity_mean',
                                                                        'return_turbidity_mean',
                                                                        'return_flow_mean',
                                                                        'tank_level_pre_rinse_mean',
                                                                        'tank_level_caustic_mean',
                                                                        'tank_level_acid_mean',
                                                                        'tank_level_clean_water_mean',
                                                                        'tank_temperature_pre_rinse_mean',
                                                                        'tank_temperature_caustic_mean',
                                                                        'tank_temperature_acid_mean',
                                                                        # 'supply_flow_sem',
                                                                        # 'supply_pressure_sem',
                                                                        # 'return_temperature_sem',
                                                                        # 'return_conductivity_sem',
                                                                        # 'return_turbidity_sem',
                                                                        # 'return_flow_sem',
                                                                        # 'tank_level_pre_rinse_sem',
                                                                        # 'tank_level_caustic_sem',
                                                                        # 'tank_level_acid_sem',
                                                                        # 'tank_level_clean_water_sem',
                                                                        # 'tank_temperature_pre_rinse_sem',
                                                                        # 'tank_temperature_caustic_sem',
                                                                        # 'tank_temperature_acid_sem',
                                                                        'supply_flow_std',
                                                                        'supply_pressure_std',
                                                                        'return_temperature_std',
                                                                        'return_conductivity_std',
                                                                        'return_turbidity_std',
                                                                        'return_flow_std',
                                                                        'tank_level_pre_rinse_std',
                                                                        'tank_level_caustic_std',
                                                                        'tank_level_acid_std',
                                                                        'tank_level_clean_water_std',
                                                                        'tank_temperature_pre_rinse_std',
                                                                        'tank_temperature_caustic_std',
                                                                        'tank_temperature_acid_std',
                                                                        'supply_flow_max',
                                                                        'supply_pressure_max',
                                                                        'return_temperature_max',
                                                                        'return_conductivity_max',
                                                                        'return_turbidity_max',
                                                                        'return_flow_max',
                                                                        'tank_level_pre_rinse_max',
                                                                        'tank_level_caustic_max',
                                                                        'tank_level_acid_max',
                                                                        'tank_level_clean_water_max',
                                                                        'tank_temperature_pre_rinse_max',
                                                                        'tank_temperature_caustic_max',
                                                                        'tank_temperature_acid_max',
                                                                        'supply_flow_min',
                                                                        'supply_pressure_min',
                                                                        'return_temperature_min',
                                                                        'return_conductivity_min',
                                                                        'return_turbidity_min',
                                                                        'return_flow_min',
                                                                        'tank_level_pre_rinse_min',
                                                                        'tank_level_caustic_min',
                                                                        'tank_level_acid_min',
                                                                        'tank_level_clean_water_min',
                                                                        'tank_temperature_pre_rinse_min',
                                                                        'tank_temperature_caustic_min',
                                                                        'tank_temperature_acid_min',
                                                                        ]))
  new_feat.drop(['1'], inplace=True, axis=1)
  new_feat['process_id'] = df.index
  new_feat = new_feat.set_index('process_id')
  return pd.concat([df, new_feat], axis=1)


def polyfeatures_all(df, p):
  new_features = df
  pol = PolynomialFeatures(p)
  new_featuresf = pol.fit_transform(new_features)
  new_feat = pd.DataFrame(new_featuresf)
  new_feat.drop(['1'], inplace=True, axis=1)
  new_feat['process_id'] = df.index
  new_feat = new_feat.set_index('process_id')
  return pd.concat([df, new_feat], axis=1)


def phase_data(df):
  def kurt(x):
    return x.kurt()
  df['return_flow__return_turbidity'] = df['return_flow'] * \
      df['return_turbidity']
  df['return_flow__return_temperature'] = df['return_flow'] * \
      df['return_temperature']
  df['return_flow__return_conductivity'] = df['return_flow'] * \
      df['conductivity']
  df['object_residue'] = df['supply_flow'] - df['return_flow']

  df['supply_flow_*_supply_pressure'] = df['supply_flow'] * \
      df['supply_pressure']
  df['supply_flow_*_return_conductivity'] = df['supply_flow'] * \
      df['return_conductivity']
  df['supply_flow_*_return_turbidity'] = df['supply_flow'] * \
      df['return_turbidity']
  df['supply_flow_*_return_temperature'] = df['supply_flow'] * \
      df['return_temperature']

  df['conductivity_*_return_temperature'] = df['conductivity'] * \
      df['return_temperature']
  df['conductivity_*_return_turbidity'] = df['conductivity'] * \
      df['return_turbidity']
  df['return_temperature_*_return_turbidity'] = df['return_temperature'] * \
      df['return_turbidity']

  tmp = df[pd_cols].groupby(['process_id', 'phase']).agg(
      ['min', 'max', 'mean', 'std', 'sum', 'median', 'skew', kurt])
  tmp = tmp.unstack(level=-1)
  tmp = tmp.fillna(0)

  # ttmp=df[['process_id','phase','return_flow','return_turbidity']].groupby(['process_id', 'phase']).apply(lambda x: pd.Series(
  #   {'sum':(x['return_flow']*x['return_turbidity']).sum(), 'mean':(x['return_flow']*x['return_turbidity']).mean(),'std':(x['return_flow']*x['return_turbidity']).std()}))
  # ttmp = ttmp.unstack(level=-1)
  # ttmp = ttmp.fillna(0)
  # ttmp = ttmp.rename(columns={'acid':'acid_sum','caustic':'caustic_sum','intermediate':'intermediate_sum','pre_rinse':'pre_rinse_sum'})
  return tmp


def prep_metadata(df):
  # select process_id and pipeline
  meta = df[['process_id', 'pipeline']
            ].drop_duplicates().set_index('process_id')

  # convert categorical pipeline data to dummy variables
  meta = pd.get_dummies(meta)

  # pipeline L12 not in test data
  if 'L12' not in meta.columns:
    meta['pipeline_L12'] = 0

  return meta


def prep_boolean(df):
  bool_cols = df.dtypes[df.dtypes == bool].index.tolist()
  boolean = df[['process_id', 'phase'] + bool_cols + ['tank_lsh_acid',
                                                      'tank_lsh_pre_rinse']]
  boolean = boolean * 1
  bl_features = boolean.groupby(['process_id', 'phase']).agg(['mean'])
  bl_features = bl_features.unstack(level=-1)
  bl_features = bl_features.fillna(0)
  # bl_features = bl_features.set_index('process_id')
  return bl_features


def prep_time_series_by_days(df, num, columns=None):
  if columns is None:
    columns = df.columns

  ts_df = df[ts_cols]

  def mean10(x):
    return x.tail(num).mean()

  def max10(x):
    return x.tail(num).max()

  def min10(x):
    return x.tail(num).min()

  def std10(x):
    return x.tail(num).std()

  def sum10(x):
    return x.tail(num).sum()

  def median10(x):
    return x.tail(num).median()

  def skew10(x):
    return x.tail(num).skew()

  def mad10(x):
    return x.tail(num).mad()

  def kurt10(x):
    return x.tail(num).kurt()

  def var10(x):
    return x.tail(num).var()

  def sem10(x):
    return x.tail(num).sem()

  ts_features = ts_df.groupby('process_id').agg(
      [mean10, max10, min10, std10, sum10, median10])
  ts_features = ts_features.rename(
      columns={
          'mean10': 'mean_%d' % num,
          'max10': 'max_%d' % num,
          'min10': 'min_%d' % num,
          'std10': 'std_%d' % num,
          'sum10': 'sum_%d' % num,
          'median10': 'median_%d' % num,
          # 'skew10': 'skew_%d' % num,
          # 'mad10': 'mad_%d' % num,
          # 'kurt10': 'kurt_%d' % num,
          # 'var_10': 'var_%d' % num,
          # 'sem_10': 'sem_%d' % num
      })
  return ts_features


def prep_time_series_features(df, columns=None):
  if columns is None:
    columns = df.columns

  def kurt(x):
    return x.kurt()

  ts_df = df[ts_cols]
  ts_df['return_flow__return_turbidity'] = df['return_flow'] * \
      df['return_turbidity']
  ts_df['return_flow__return_temperature'] = df['return_flow'] * \
      df['return_temperature']
  ts_df['return_flow__return_conductivity'] = df['return_flow'] * \
      df['return_conductivity']
  ts_df['object_residue'] = ts_df['supply_flow'] - ts_df['return_flow']
  ts_features = ts_df.groupby('process_id').agg(
      ['min', 'max', 'std', 'mean', 'sum', 'median'])

  return ts_features


def prep_time_series_features_min(df, columns=None):
  if columns is None:
    columns = df.columns

  ts_df = df[ts_cols]
  ts_features = ts_df.groupby('process_id').agg(
      ['min', 'max', 'mean'])

  return ts_features


def prep_spent_time(df):
  timespent = df[['process_id', 'timestamp', 'phase']].groupby(
      ['process_id', 'phase']).agg(['count']).unstack(level=1).fillna(0) * 2
  return timespent


def create_feature_matrix(df):
  # metadata = prep_metadata(df)
  time_series = prep_time_series_features(df)
  print "time_series done ....."
  # bool_vals = prep_boolean(df)
  phase_vals = phase_data(df)
  print "phase_vals done ....."
  spent_time = prep_spent_time(df)
  print "spent_time done ....."
  pipeline_time = create_pipeline_time(df)
  print "pipeline_time done ....."
  # object_ids = prep_object(df)
  time_series_5 = prep_time_series_by_days(df, 5)
  print "time_series_5 done ....."
  time_series_10 = prep_time_series_by_days(df, 10)
  print "time_series_10 done ....."
  time_series_50 = prep_time_series_by_days(df, 50)
  print "time_series_50 done ....."
  time_series_100 = prep_time_series_by_days(df, 100)
  print "time_series_100 done ....."
  # time_series_500 = prep_time_series_by_days(df, 500)
  # print "time_series_500 done ....."

  # object_ts = prep_object_data(df)
  # join metadata and time series features into a single dataframe
  feature_matrix = pd.concat(
      [time_series, time_series_5, time_series_10, time_series_50, time_series_100, phase_vals, spent_time, pipeline_time], axis=1)
  return feature_matrix


def create_min_feature_matrix(df):
  # metadata = prep_metadata(df)
  time_series = prep_time_series_features_min(df)
  # bool_vals = prep_boolean(df)
  # phase_vals = phase_data(df)
  # spent_time = prep_spent_time(df)
  # pipeline_time = create_pipeline_time(df)
  # join metadata and time series features into a single dataframe
  # feature_matrix = pd.concat(
  #     [metadata, time_series], axis=1)
  return time_series


def create_pipeline_time(df):
  t_spent = prep_spent_time(df)
  t_spent['pipeline'] = df[['process_id', 'pipeline']
                           ].drop_duplicates().set_index('process_id')
  t_spent = t_spent.groupby(by='pipeline', as_index=False).agg([
      'min', 'max', 'std', 'mean'])
  t_spent.columns = t_spent.columns.map('_'.join)
  t_spent = df.join(on='pipeline', other=t_spent)[
      t_spent.columns.tolist() + ['process_id']]
  t_spent = t_spent.drop_duplicates().set_index('process_id')
  t_spent = t_spent.fillna(0)
  return t_spent


def prep_object(df):
  df['object_id'] = 'object_' + df['object_id'].astype(str)
  # top50 = df[['process_id', 'object_id']].drop_duplicates().object_id.value_counts()[
  #     :50].index.tolist()
  print top50
  tmpdf = df[['process_id', 'object_id']
             ].drop_duplicates().set_index('process_id')
  tmpdf.loc[tmpdf[~tmpdf.object_id.isin(
      top50)]['object_id'].index.tolist(), 'object_id'] = 'object_other'
  return pd.get_dummies(tmpdf)


def prep_object_data(df):
  tmp = df[data_cols].groupby(['object_id'], as_index=False).agg(
      ['min', 'max', 'std', 'mean'])
  tmm = df[['process_id', 'object_id']].drop_duplicates()
  tmp.columns = tmp.columns.map('__'.join)
  tmp = tmm.join(on='object_id', other=tmp)[
      tmp.columns.tolist() + ['process_id']]
  tmp = tmp.set_index('process_id')
#     tmp = tmp.unstack(level=-1)
#     tmp = tmp.fillna(0)
#     tmp = tmp.set_index(tmp.index.get_level_values(0))
  return tmp


def prep_categorical(train, test):
  train['object_id'] = 'object_' + train['object_id'].astype(str)

  tmpdf = train[['process_id', 'object_id']
                ].drop_duplicates().set_index('process_id')
  meta = train[['process_id', 'pipeline']
               ].drop_duplicates().set_index('process_id')

  from sklearn import preprocessing
  le_pipe = preprocessing.LabelEncoder()
  le_pipe.fit(meta['pipeline'])
  labels_pipe = le_pipe.transform(meta['pipeline'])

  le_obj = preprocessing.LabelEncoder()
  le_obj.fit(tmpdf['object_id'])
  labels_obj = le_obj.transform(tmpdf['object_id'])

  train_cat = tmpdf = train[['process_id']
                            ].drop_duplicates().set_index('process_id')
  train_cat['object_id'] = labels_obj
  train_cat['pipeline'] = labels_pipe

  test['object_id'] = 'object_' + test['object_id'].astype(str)

  tmpdf_test = test[['process_id', 'object_id']
                    ].drop_duplicates().set_index('process_id')
  meta_test = test[['process_id', 'pipeline']
                   ].drop_duplicates().set_index('process_id')
  labels_pipe_test = le_pipe.transform(meta_test['pipeline'])
  labels_obj_test = le_obj.transform(tmpdf_test['object_id'])

  test_cat = tmpdf = test[['process_id']
                          ].drop_duplicates().set_index('process_id')
  test_cat['object_id'] = labels_obj_test
  test_cat['pipeline'] = labels_pipe_test

  return train_cat, test_cat


def get_operating_time(df):
  tss = df[['timestamp', 'process_id']].groupby(
      'process_id').agg(['first', 'last'])
  tss['start_hour'] = tss[('timestamp', 'first')].dt.hour
  tss['start_min'] = tss[('timestamp', 'first')].dt.minute
  tss['end_hour'] = tss[('timestamp', 'last')].dt.hour
  tss['end_min'] = tss[('timestamp', 'last')].dt.minute
  tss['dayofweek'] = tss[('timestamp', 'last')].dt.dayofweek
  tss_new = df[['process_id']
               ].drop_duplicates().set_index('process_id')
  tss_new['start_hr_sin'] = np.sin(tss.start_hour * (2. * np.pi / 24))
  tss_new['start_hr_cos'] = np.cos(tss.start_hour * (2. * np.pi / 24))
  tss_new['end_hr_sin'] = np.sin(tss.end_hour * (2. * np.pi / 24))
  tss_new['end_hr_cos'] = np.cos(tss.end_hour * (2. * np.pi / 24))

  tss_new['start_min_sin'] = np.sin(tss.start_min * (2. * np.pi / 60))
  tss_new['start_min_cos'] = np.cos(tss.start_min * (2. * np.pi / 60))
  tss_new['end_min_sin'] = np.sin(tss.end_min * (2. * np.pi / 60))
  tss_new['end_min_cos'] = np.cos(tss.end_min * (2. * np.pi / 60))

  tss_new['dayofweek_sin'] = np.sin((tss.dayofweek) * (2. * np.pi / 7))
  tss_new['dayofweek_cos'] = np.cos((tss.dayofweek) * (2. * np.pi / 7))
#     tss = tss.drop(['start_hour','start_min','end_hour','end_min','dayofweek',('timestamp','first'),('timestamp','last')],axis=1)
#     tss = tss.unstack(level=-1)
  return tss_new
