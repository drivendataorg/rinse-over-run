import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

import cPickle as pickle
DATA_DIR = './data/processed/features/'


# from src.visualization.visualize import plot_feature_importances,plot_feature_importances_cum
from src.features.load_features import load_train_test, load_train_min_test, load_train_outlier_test, load_train_outlier_plus_test
from src.features.build_features import polyfeatures, polyfeatures_all
from sklearn.model_selection import KFold
import gc
from sklearn import preprocessing

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, KFold


def MAPE(y_true, y_pred):
    deno = np.copy(y_true)
    deno[np.abs(deno) < 290000] = 290000.0
    return np.average(np.abs(y_pred - y_true) / deno)


def lgbm_mape(preds, train_data):
    '''
    Custom Evaluation Function for LGBM
    '''
    labels = train_data.get_label()
    smape_val = MAPE(labels, preds)
    return 'SMAPE', smape_val, False


def kfold_train(params, data_, test_, y_, folds_, feats=None, nrounds=50000, random_state=42, model_name='model', categorical_feature='auto'):
    oof_preds = np.zeros(data_.shape[0])
    sub_preds = np.zeros(test_.shape[0])
    gbm = ''
    gbms = []
    feature_importance_df = pd.DataFrame()
    if feats is None:
        feats = data_.columns
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_)):
        evals_result = {}
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_[val_idx]
        lgb_train = lgb.Dataset(
            trn_x, trn_y, categorical_feature=categorical_feature)
        lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)
        params['seed'] = params['seed'] + n_fold
        #del gbm
        gbm = lgb.train(params, lgb_train, nrounds, valid_sets=lgb_eval, feval=lgbm_mape, evals_result=evals_result,
                        verbose_eval=50, early_stopping_rounds=300,)

        oof_preds[val_idx] = gbm.predict(
            val_x, num_iteration=gbm.best_iteration)
        sub_preds += gbm.predict(test_[feats],
                                 num_iteration=gbm.best_iteration) / folds_.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = trn_x.columns
        fold_importance_df["importance"] = gbm.feature_importance(
            importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d Training MAPE : %.6f' % (
            n_fold + 1, MAPE(trn_y, gbm.predict(trn_x, num_iteration=gbm.best_iteration))))
        print('Fold %2d Testing MAPE : %.6f' %
              (n_fold + 1, MAPE(val_y, oof_preds[val_idx])))
        del trn_x, trn_y, val_x, val_y
        gbms.append(gbm)
        gc.collect()

    print('Full MAPE score %.6f' % MAPE(y, oof_preds))

    test_['final_rinse_total_turbidity_liter'] = sub_preds

    return oof_preds, test_[['final_rinse_total_turbidity_liter']], feature_importance_df, gbms


mape_params = {
    'learning_rate': 0.01,
    'boosting_type': 'dart',
    'objective': 'mape',
    'sub_feature': 0.7,
    'lambda_l1': 5e-3,
    'lambda_l2': 5e-3,
    'min_gain_to_split': 5e-6,
    'num_leaves': 100,
    'min_data': 40,
    'min_hessian': 1,
    'verbose': -1,
    'seed': 314,
    'max_bin': 150,
    'reg_sqrt': True,
    "boost_from_average": True,
    'tree_learner': 'feature',
    'min_data_per_group': 50,
    'metric': "mape",
    'zero_as_missing': True,
    'is_training_metric': True,
    'num_threads': 8

}
training_param = {
    'timeseries': [10, 50, 200, 500],
    'time': True,
    'pipeline_time': False,
    'phase_data': True,
    'categorical': True,
    'object_data': False,
    'ts_all': False,
    'boolean_data': False,
    'operating': True

}
train_features, test_features = load_train_test(
    DATA_DIR=DATA_DIR, **training_param)
training_labels = pd.read_csv(
    DATA_DIR + '../../raw/train_labels.csv', index_col=0)
cleaning_recipe = pd.read_csv(
    DATA_DIR + '../../raw/recipe_metadata.csv', index_col=0)
cleaning_recipe['combination'] = cleaning_recipe['pre_rinse'].astype(str) + cleaning_recipe['caustic'].astype(
    str) + cleaning_recipe['intermediate_rinse'].astype(str) + cleaning_recipe['acid'].astype(str) + cleaning_recipe['final_rinse'].astype(str)
recipe = preprocessing.LabelEncoder()
recipe.fit(cleaning_recipe['combination'])
cleaning_recipe['combination_cat'] = recipe.transform(
    cleaning_recipe['combination'])
cleaning_recipe_train = cleaning_recipe['combination_cat'].loc[train_features.index.unique(
)]
train_features = pd.concat([train_features, cleaning_recipe_train], axis=1)
cleaning_recipe_test = cleaning_recipe['combination_cat'].loc[test_features.index.unique(
)]
test_features = pd.concat([test_features, cleaning_recipe_test], axis=1)
train_labels = training_labels.loc[train_features.index.unique()]


featt = pd.read_csv('src/models/feat.csv')
ft = featt.head(550)['feature'].tolist()
ff = []
for ee in ft:
    try:
        ff.append(eval(ee))
    except:
        ff.append(ee)
train_features_red = train_features.iloc[:, train_features.columns.isin(ff)]
test_features_red = test_features.iloc[:, test_features.columns.isin(ff)]

train_labels = train_labels.loc[train_features.index.unique()]
train_labels['log'] = np.log(train_labels)
train_labels_out = train_labels[(train_labels['log'] > train_labels['log'].mean(
) - 2.5 * train_labels['log'].std()) & (train_labels['log'] < train_labels['log'].mean() + 2.5 * train_labels['log'].std())]
train_features_new = train_features_red.loc[train_labels_out.index.unique()]

gc.enable()
# Build model inputs
data, test, y = train_features_new, test_features_red, np.ravel(
    train_labels_out['final_rinse_total_turbidity_liter'])
#data, test, y = train_features_red,test_features_red,np.ravel(train_labels.loc[train_features.index.unique()])
# Create Folds
folds = KFold(n_splits=10, shuffle=True, random_state=5156)
# Train model and get oof and test predictions
oof_preds, test_preds_0, importances, last_gbm = kfold_train(
    mape_params, data, test, y, folds, categorical_feature=['pipeline', 'object_id', 'combination_cat'])
# Save test predictions
# test_preds.to_csv('submission_test.csv')
# Display a few graphs
folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(data)]

test_preds_0.to_csv('./models/submission_test.csv')
