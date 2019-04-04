import pandas as pd
import numpy as np
import feather
import lightgbm as lgb
from sklearn.model_selection import RepeatedKFold
from tqdm import tqdm
import pickle

# Custom loss function
def rinse_mape(y, preds):
    return np.mean(abs(y-preds)/np.maximum(y,290000))

def rinse_mape_lgb(preds, train_data):
    y = train_data.get_label()
    return 'loss', rinse_mape(y, preds), False

# Fit lightGBM model
def fit_lgb(train, y, cat_cols, params, random_state=1, nfolds=5, n_repeats=10):
    # Get optimal number of iterations
    cv = lgb.cv(
        params,
        stratified=False,
        train_set=lgb.Dataset(train, label=y, categorical_feature=cat_cols),
        nfold=nfolds,
        num_boost_round=20000,
        early_stopping_rounds=100,
        verbose_eval=-1,
        feval=rinse_mape_lgb)
    
    best_iter = np.argmin(cv['loss-mean'])
    best_score = round(cv['loss-mean'][best_iter],2)
    print(f'Best number of iterations: {best_iter}, score: {best_score}')
    
    # Fit
    print(f'{nfolds}-fold cross-validation, {n_repeats} repeats:')
    val_pred = np.empty((len(train),nfolds*n_repeats))
    val_pred[:] = np.nan
    folds = RepeatedKFold(n_splits=nfolds, 
                          n_repeats=n_repeats,
                          random_state=random_state).split(y)
    models = list()
    for i, (trn_idx, val_idx) in enumerate(folds):
        print('Fit fold', i+1, 'out of', nfolds*n_repeats)

        trn_X, val_X  = train.iloc[trn_idx], train.iloc[val_idx]
        trn_y, val_y = y[trn_idx], y[val_idx]

        trn_lgb = lgb.Dataset(trn_X, label=trn_y, categorical_feature=cat_cols)
        val_lgb = lgb.Dataset(val_X, label=val_y, categorical_feature=cat_cols)

        m = lgb.train(params,
                      trn_lgb,
                      num_boost_round = best_iter,
                      valid_sets = [trn_lgb, val_lgb],
                      verbose_eval=-1,
                      feval=rinse_mape_lgb)
        
        val_pred[val_idx, i] = m.predict(val_X)
        models.append(m)
    
    val_pred = np.nanmean(val_pred, axis=1)
    val_df = pd.DataFrame(data = {'target': y, 'yhat': val_pred},
                          index=train.index)
    print('Validation loss:', rinse_mape(y, val_pred))
    return models, val_df

def predict_lgb(test, models, features=None):
    if features==None: features=test.columns
    test_pred = np.zeros((len(test),))
    for m in models:
        test_pred += m.predict(test[features])/len(models)    
    sub_df = pd.DataFrame(data = test_pred,
                      index=test.index, 
                      columns=['final_rinse_total_turbidity_liter'])
    return sub_df

# lightGBM parameters
params = {
    'boosting_type': 'dart',
    'objective': 'regression_l1',
    'num_leaves': 137,
    'metric': 'lgb_final_mape',
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5410338291948477,
    'bagging_freq': 0,
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'max_depth': 16,
    'min_child_weight': 83.09368167218351,
    'min_split_gain': 0.09999999999999999,
    'verbose': -1
}

# Load data
phase_list = ['pre_rinse', 'caustic', 'intermediate_rinse', 'acid']
all_train = feather.read_dataframe('tmp/train.feather')
all_test = feather.read_dataframe('tmp/test.feather')

# Encode categorricals as int for lightGBM
all_cat_cols = all_train.columns[all_train.dtypes == 'category'].tolist()
for c in all_cat_cols:
    all_train[c] = all_train[c].cat.codes
    all_test[c] = all_test[c].cat.codes

# Create feature list, load selected features for each phase
feature_cols = [c for c in all_train.columns.tolist() if c not in 
                ['process_id', 'target_time_period',
             'orig_process_id', 'max_phase', 'last_phase', 'target']]
feature_dict = pickle.load(open('selected_features.pickle', 'rb'))

# Fit by phase
phase_sub = list()
loss_list = list()

for i,p in enumerate(tqdm(phase_list)):
    # Select features for each phase
    feature_cols = feature_dict[p]
    cat_cols = [c for c in all_cat_cols if c in feature_cols]
    
    # Use all observations for 'acid', for other phases use only observations ending at that phase 
    train = all_train[all_train['max_phase']==p]
    if p != 'acid': train = train[train['last_phase']==p]

    # List of observations to use for validation loss later
    phase_idx = train['last_phase']==p 
    
    # Estimate
    train = train[feature_cols]
    y = all_train.loc[train.index, 'target'].values
    models, val = fit_lgb(train, y, cat_cols, params, random_state=1, nfolds=5, n_repeats=10)
    
    # Calculate cross-validation loss
    loss_list.append(rinse_mape(val.loc[phase_idx, 'target'], 
                                val.loc[phase_idx, 'yhat']))
    
    # Predict
    test = all_test.loc[all_test['last_phase']==p, feature_cols]
    sub = predict_lgb(test, models)
    phase_sub.append(sub)

for i,p in enumerate(phase_list):
    print(p, round(loss_list[i],4))
print('Validation loss:', round(np.dot(loss_list, [0.1,0.3,0.3,0.3]),4))

# Create submission file
test_pred = pd.concat(phase_sub).sort_index()
sub = pd.read_csv('data/submission_format.csv')
sub['final_rinse_total_turbidity_liter'] = test_pred['final_rinse_total_turbidity_liter']
sub.to_csv('output/sub.csv', index=False)