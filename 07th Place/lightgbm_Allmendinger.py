#
# TODO:
# create model for each phase
#
# Competion: Sustainable Industry: Rinse Over Run
# Author: Bernd Allmendinger
# Date : 02.03.2019
#
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

pd.set_option('display.max_columns', 40)


Version = 27
T = 290000.0
np.random.seed(2018)

def lgb_mean_absolute_max_error(preds, train_data):
    labels = np.expm1(train_data.get_label())
    d=np.maximum(np.abs(labels), np.array([T]*len(labels)))
    return 'mame', np.mean(np.abs(labels - np.expm1(preds)) / d), False

def mean_absolute_max_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    d=np.maximum(np.abs(y_true), np.array([T]*len(y_true)))
    return np.mean(np.abs(y_true - y_pred) / d)

DATA_DIR = Path('Data')

submission_format = pd.read_csv(DATA_DIR / 'submission_format.csv')
submission_format=submission_format.drop(['final_rinse_total_turbidity_liter'], axis=1)

# variables we'll use to create our time series features
ts_cols = [
    'process_id',
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
    'tank_lsh_clean_water',
    'tank_lsh_pre_rinse',
    'Vol_turbidity',
    'Vol_conductivity',
]

ts_bool_cols = [
    'process_id',
    'supply_pump',
    'supply_pre_rinse',
    'supply_caustic',
    'return_caustic',
    'supply_acid',
    'return_acid',
    'supply_clean_water',
    'return_recovery_water',    
    'return_drain',
    'object_low_level',
    'tank_lsh_caustic',
    'tank_lsh_clean_water'
]


recipe_metadata = pd.read_csv(DATA_DIR / 'recipe_metadata.csv')
recipe_metadata=recipe_metadata.drop(['final_rinse'], axis=1)

phase_map={'pre_rinse' : 1, 'caustic': 2, 'intermediate_rinse' : 3, 'acid': 4}

# load train data
train_values = pd.read_csv(DATA_DIR / 'train_values.csv',  index_col=0,  parse_dates=['timestamp'])
train_labels = pd.read_csv(DATA_DIR / 'train_labels.csv',  index_col=0)
#train_values = pd.concat([train_values, recipe_metadata], axis=1)
train_values = train_values.merge(recipe_metadata, how='left',  on='process_id') 


# load the test data
test_values = pd.read_csv(DATA_DIR / 'test_values.csv',   index_col=0, parse_dates=['timestamp'])
test_values['phase'] =test_values['phase'].map(phase_map)
#test_values = pd.concat([test_values, recipe_metadata], axis=1)
test_values = test_values.merge(recipe_metadata, how='left',  on='process_id') 

# Calculate  maxe phase
test_phase=test_values.groupby('process_id')['phase'].max().reset_index()
test_phase=test_phase.rename(columns={"phase": "maxphase"})
 
test_values=test_values.reset_index()
test_values=test_values.merge(test_phase, how='left',  on='process_id') # nach disem wert wird speraiert und die prgonosen jeweils mit einem eigenen modell gemacht
   
train_values = train_values[train_values.phase != 'final_rinse']
train_values['phase'] =train_values['phase'].map(phase_map)

del recipe_metadata

# convert boo, to integer
for c in ts_bool_cols:
    train_values[c] = train_values[c].astype(int)
    test_values[c] = test_values[c].astype(int)


def prep_metadata(df):
    X = df.copy().set_index('process_id')
    meta=X.groupby('process_id')['object_id', 'pipeline', 'pre_rinse', 'caustic', 'intermediate_rinse', 'acid'].head(1)
  
    pipe_map={"L"+str(i+1) :i+1  for i in range(12)}
    meta['pipeline'] =meta['pipeline'].map(pipe_map)
    
    meta['num_phases_recepie']= meta['pre_rinse']+meta['caustic']+meta['intermediate_rinse']+meta['acid']
    # calculate number of phases for each process_object
    meta['num_phases'] = df.groupby('process_id')['phase'].apply(lambda x: x.nunique())
    return meta


def prep_time_series_features(df, maxphase):
    df_feature = pd.DataFrame({'process_id':df.process_id.unique()})
    df_feature = df_feature.set_index('process_id')
    
    # for phase
    colL = []
    if maxphase==1:
        phaseL = [maxphase]
    elif maxphase <=3:
        phaseL = [maxphase-1, maxphase]
    else:
        phaseL = [1,2,3,4]
    # for phase in np.arange(maxphase)+1:
    for phase in phaseL:
        dfp = df[df.phase == phase].copy()
        
        # fload features
        ts_df = dfp[ts_cols].set_index('process_id')
        ts_features = ts_df.groupby('process_id').agg(['min', 'max', 'mean', 'std', 'count', lambda x: x.tail(5).mean()])
        ts_features.columns = ['P'+str(phase)+ '_'+'_'.join(col).strip() for col in ts_features.columns.values]
        df_feature = pd.concat([df_feature, ts_features], axis=1)
        colL.append(ts_features.columns.values)
        # features für diff tail and head
        
        ts_tail = ts_df.groupby('process_id').agg([lambda x: x.tail(5).mean()])
        ts_tail.columns = ['Diff'+str(phase)+ '_'+'_'.join(col).strip() for col in ts_tail.columns.values]
        
        ts_head = ts_df.groupby('process_id').agg([lambda x: x.head(5).mean()])
        ts_head.columns = ['Diff'+str(phase)+ '_'+'_'.join(col).strip() for col in ts_head.columns.values]
        
        for col in ts_tail.columns:
            ts_tail[col] = ts_tail[col]-ts_head[col]
        
        df_feature = pd.concat([df_feature, ts_tail], axis=1)
        
        # Bool features
        dfp = df[df.phase == phase].copy()
        ts_df = dfp[ts_bool_cols].set_index('process_id')
        ts_features = ts_df.groupby('process_id').agg(['sum', lambda x: x.tail(5).sum()])
        ts_features.columns = ['Bool_P'+str(phase)+ '_'+'_'.join(col).strip() for col in ts_features.columns.values]
        df_feature = pd.concat([df_feature, ts_features], axis=1)

    
    if len(colL)>1:
        for (col1, col2) in zip(colL[-1],colL[-2]): # letzter eintrag in Liste
            c=col1[3:]
            p1=col1[1:2]
            p2=col2[1:2]
            df_feature['Phasediff'+c+'_'+p1+p2] = df_feature[col1]-df_feature[col2]
    

    df_feature=df_feature.fillna(0)    
    return df_feature


# Function to aggregate all feature engineering.
def create_feature_matrix(df, maxphase):
    df['Vol_turbidity'] =  df.apply(lambda row: row['return_flow']*row['return_turbidity'] if row['return_flow']>0.0 else 0.0, axis=1)
    df['Vol_conductivity'] =  df.apply(lambda row: row['return_flow']*row['return_conductivity'] if row['return_flow']>0.0 else 0.0, axis=1)

    metadata = prep_metadata(df)
    time_series = prep_time_series_features(df, maxphase)
    
    # join metadata and time series features into a single dataframe
    feature_matrix = pd.concat([metadata, time_series], axis=1)
    
    return feature_matrix


params ={
 'boosting_type': 'gbdt',  
 'learning_rate': 0.01, 
 'objective': 'mape', 
 'num_leaves': 49,
 'bagging_freq': 5, 
 'min_child_samples': 23, 
 'min_data_in_leaf': 15, 
 'feature_fraction': 0.6, 
 'bagging_fraction': 0.9, 
 'min_data_per_group': 10,
 'max_cat_threshold ': 32, 
 'min_sum_hessian_in_leaf   ': 0.01
 }


totalscore=[] 
for phase in np.arange(4)+1:

    X = train_values[train_values.phase <= phase].copy()
    x_train = create_feature_matrix(X, phase)
    x_train.head()
    
    x_train =  x_train.reset_index().merge(train_labels.reset_index(), how='left',  on='process_id')
    y_train = x_train['final_rinse_total_turbidity_liter']
    x_train=x_train.drop(['process_id', 'final_rinse_total_turbidity_liter'], axis=1)
    
    dtrain = lgb.Dataset(x_train,  np.log1p(y_train), free_raw_data=False, categorical_feature=[0,1])
    gbm = lgb.cv(params,
                    early_stopping_rounds=50,
                    nfold =5,
                    stratified=False,
                    verbose_eval=100,
                    feval=lgb_mean_absolute_max_error,
                    train_set =dtrain,
                    num_boost_round=1000000)
    

    best_rounds=np.argmin(gbm['mame-mean'])+1
    best_score = np.min(gbm['mame-mean'])
    totalscore.append(best_score)
    print ("phase %i best score %8.4f best round %i" % (phase, best_score, best_rounds))
    
    ##############################
    gbm = lgb.train(params, dtrain,  num_boost_round=best_rounds) 
    plot=lgb.plot_importance(gbm, max_num_features=50, figsize=(8, 12))
        
    # create metadata and time series features
    test_values_p = test_values[test_values.maxphase == phase].copy() # mach prediction nur für die phase 
    x_test = create_feature_matrix(test_values_p, phase)
    del test_values_p
    
    preds = np.expm1(gbm.predict(x_test)) # prediction für phase=phase
    x_test=x_test.reset_index()
    preds= pd.DataFrame({'process_id': x_test.process_id, 'final_rinse_total_turbidity_liter': preds})
    if phase==1:
        df = preds
    else:
        df=df.append(preds, ignore_index=True)

submission_format=submission_format.merge(df, how='left',  on='process_id') 
    
cv_total_score = 0.1*totalscore[0]+0.3*totalscore[1]+0.3*totalscore[2]+0.3*totalscore[3]
print("Total lgb cv: ", cv_total_score)      

submission_format.head()

submission_format.to_csv('submission_GMBlight_'+str(Version) +'.csv', index=False)


'''
Results:
phase 1 best score   0.2941 best round 1018
phase 2 best score   0.2748 best round 907
phase 3 best score   0.2638 best round 1160
phase 4 best score   0.2540 best round 1126
'Total lgb cv: ', 0.267187

'''