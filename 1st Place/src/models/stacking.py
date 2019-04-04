# The essentials
import pandas as pd
import numpy as np

# CLI & Logging
import click
import logging

# A whole bunch of sklearn classifiers & utils
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.base import clone
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.preprocessing import StandardScaler

# Ignore any warnings
import warnings
warnings.filterwarnings('ignore')

clfs = [
    ('knn_100', Pipeline(steps=[
        ('scale', StandardScaler()), 
        ('knn_100', KNeighborsRegressor(n_neighbors=100))
    ])),
    ('knn_10', Pipeline(steps=[
        ('scale', StandardScaler()), 
        ('knn_10', KNeighborsRegressor(n_neighbors=10))
    ])),
    ('knn_50', Pipeline(steps=[
        ('scale', StandardScaler()), 
        ('knn_50', KNeighborsRegressor(n_neighbors=50))
    ])),
    ('knn_250', Pipeline(steps=[
        ('scale', StandardScaler()), 
        ('knn_250', KNeighborsRegressor(n_neighbors=250))
    ])),
    ('lr', Pipeline(steps=[
        ('scale', StandardScaler()), 
        ('lr', Lasso(max_iter=1000))
    ])),
    ('knn_pca', Pipeline(steps=[
        ('scale', StandardScaler()), 
        ('pca', PCA(n_components=5)), 
        ('knn', KNeighborsRegressor(n_neighbors=100))
    ])),
    ('svr', Pipeline(steps=[
        ('scale', StandardScaler()), 
        ('svr', SVR())
    ])),
    ('linear_svr', Pipeline(steps=[
        ('scale', StandardScaler()), 
        ('svr', LinearSVR())
    ])),
    ('rf_25', RandomForestRegressor(n_estimators=25)),
    ('rf_100', RandomForestRegressor(n_estimators=100)),
    ('rf_250', RandomForestRegressor(n_estimators=250)),
    ('et_50', ExtraTreesRegressor(n_estimators=50)),
    ('et_100', ExtraTreesRegressor(n_estimators=100)),
    ('et_250', ExtraTreesRegressor(n_estimators=250)),
]

combinations_per_recipe = {
    3: [1, 3], 
    9: [8],
    15: [1, 3, 7, 15]
}


def custom_mape(approxes, targets):
    """Competition metric is a slight variant on MAPE."""
    nominator = np.abs(np.subtract(approxes, targets))
    denominator = np.maximum(np.abs(targets), 290000)
    return np.mean(nominator / denominator)


def fit_stack(clf, name, X_train, y_train, X_test, train_index, test_index, 
              n_splits=5):  
    train_predictions = np.zeros((len(X_train),))
    test_predictions = np.zeros((len(X_test), n_splits))
    kf = KFold(n_splits=n_splits, shuffle=True)
    for fold_ix, (train_idx, test_idx) in enumerate(kf.split(X_train, y_train)):
        X_cv_train = X_train[train_idx, :]
        X_cv_test = X_train[test_idx, :]
        y_cv_train = y_train[train_idx]
        y_cv_test = y_train[test_idx]

        clf_clone = clone(clf)
        clf_clone.fit(X_cv_train, y_cv_train)

        logging.info('[{}] Fold #{} MAPE={}'.format(name, fold_ix + 1, custom_mape(np.exp(y_cv_test), np.exp(clf_clone.predict(X_cv_test)))))

        train_predictions[test_idx] = np.minimum(np.max(y_cv_train), np.maximum(0, clf_clone.predict(X_cv_test)))
        test_predictions[:, fold_ix] = np.minimum(np.max(y_cv_train), np.maximum(0, clf_clone.predict(X_test)))

    train_predictions_df = pd.DataFrame(train_predictions, index=train_index, columns=['{}_pred'.format(name)])
    test_predictions_df = pd.DataFrame(np.mean(test_predictions, axis=1), index=test_index, columns=['{}_pred'.format(name)])
    
    return train_predictions_df, test_predictions_df


@click.command()
@click.argument('feature_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
def main(feature_path, output_path):
    for rec in combinations_per_recipe:
        for comb in combinations_per_recipe[rec]:
            format_str = 'Fitting classifiers for recipe {} and combination {}'
            logging.info(format_str.format(rec, comb))
            path = '{}/train_features_{}_{}.csv'.format(feature_path, rec, comb)
            train_features = pd.read_csv(path, index_col=0)
            path = '{}/test_features_{}_{}.csv'.format(feature_path, rec, comb)
            test_features = pd.read_csv(path, index_col=0)
            
            X_train = train_features.drop('target', axis=1)
            y_train = np.log(train_features['target'])
            X_test = test_features
            train_index = X_train.index
            test_index = X_test.index

            format_str = 'Train shape = {} || Test shape = {}'
            logging.info(format_str.format(X_train.shape, X_test.shape))

            for name, clf in clfs:
                train_pred_df, test_pred_df = fit_stack(clf, name, X_train.values, y_train.values, X_test.values, train_index, test_index, n_splits=10)
                path = '{}/train_predictions_stack_{}_{}_{}.csv'.format(output_path, name, rec, comb)
                train_pred_df.to_csv(path)
                path = '{}/test_predictions_stack_{}_{}_{}.csv'.format(output_path, name, rec, comb)
                test_pred_df.to_csv(path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
