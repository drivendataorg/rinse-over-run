# The essentials
import pandas as pd
import numpy as np

# CLI & Logging
import click
import logging

# Gradient Boosting
from catboost import CatBoostRegressor

# Plotting
import matplotlib.pyplot as plt

# ML utils
from sklearn.model_selection import KFold

# Some fancy printing
from terminaltables import AsciiTable

# Retrieving files from HD using regexes
import glob

# Python standard library
from collections import defaultdict
import datetime
import itertools

# Model explanation with Shapley values
import shap

combinations_per_recipe = {
    3: [1, 2, 3], 
    9: [8],
    15: [1, 2, 3, 6, 7, 14, 15]
}

weights = {
    (3, 1): 0.0219,
    (3, 2): 0.0064,
    (3, 3): 0.1695,
    (9, 8): 0.0411,
    (15, 1): 0.0765,
    (15, 2): 0.0013,
    (15, 3): 0.2289,
    (15, 6): 0.0007,
    (15, 7): 0.2258,
    (15, 14): 0.0017,
    (15, 15): 0.2262,
}


def custom_mape(approxes, targets):
    """Competition metric is a slight variant on MAPE."""
    nominator = np.abs(np.subtract(approxes, targets))
    denominator = np.maximum(np.abs(targets), 290000)
    return np.mean(nominator / denominator)


class MAPEMetric(object):
    """eval_metric for CatBoost (can only be used when task_type=CPU)."""
    def get_final_error(self, error, weight):
        return error

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, targets, weight):
        return custom_mape(np.exp(approxes), np.exp(targets)), len(targets)


def get_validation_data(X_train, y_train):
    """Just take 5% of the data at random to serve as validation set."""
    train_idx = np.random.choice(X_train.index, replace=False, 
                                 size=int(0.95 * len(X_train)))
    val_idx = list(set(X_train.index) - set(train_idx))

    X_val = X_train.loc[val_idx, :]
    y_val = y_train.loc[val_idx]
    X_train = X_train.loc[train_idx, :]
    y_train = y_train.loc[train_idx]

    return X_train, y_train, X_val, y_val


def generate_shapley(model, X_train, X_val, X_test, output_path=None):
    """Generate a Shapley plot, based on all data."""
    explainer = shap.TreeExplainer(model)
    all_data = pd.concat([X_train, X_val, X_test])
    shap_values = explainer.shap_values(all_data.values)

    plt.figure()
    shap.summary_plot(shap_values, all_data, max_display=30, 
                      auto_size_plot=True, show=False, color_bar=False)
    
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()


def fit_cat(X_train, y_train, X_test, output_path=None):
    """Fit a CatBoost Gradient Booster, plot shapley value and return 
    predictions on test set."""
    X_train, y_train, X_val, y_val = get_validation_data(X_train, y_train)

    cat = CatBoostRegressor(iterations=10000, od_type='Iter', od_wait=100, 
                            learning_rate=0.33,
                            loss_function='MAPE', eval_metric=MAPEMetric(), 
                            task_type='CPU')
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)

    predictions = cat.predict(X_test)

    generate_shapley(cat, X_train, X_val, X_test, 
                     output_path='{}_shap.png'.format(output_path))

    return predictions


def fit_cat_cv(X, y, rec, comb, output_path=None, n_folds=5):
    """Wrapper around fit_cat to generate out-of-sample predictions in CV."""
    cv_predictions = []
    kf = KFold(n_splits=n_folds, random_state=2019, shuffle=True)
    for fold_nr, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train = X.iloc[train_idx, :]
        X_test = X.iloc[test_idx, :]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        log_str = 'Fitting data from recipe {} and combination {}...'
        logging.info(log_str.format(rec, comb))
        path = '{}/{}_{}_{}'.format(output_path.rstrip('/'), rec, comb, 
                                    fold_nr + 1)
        predictions = np.exp(fit_cat(X_train, y_train, X_test, 
                                     output_path=path))
        
        mape = custom_mape(predictions, np.exp(y_test))
        log_str = '[Recipe: {} Present Phases: {} Fold: {}] TEST MAPE = {}'
        logging.info(log_str.format(rec, comb, fold_nr + 1, mape))
        
        cv_predictions.append(pd.DataFrame(np.reshape(predictions, (-1, 1)), 
                                                      index=X_test.index, 
                                                      columns=['prediction']))

    prediction_df = pd.concat(cv_predictions)
    return prediction_df


def load_data(mode, rec, comb, feature_path, stack_path):
    """Load features, optionally append with predictions from other
    classifiers for stacking. If 'target' column is present, store it
    in y."""
    path = '{}/{}_features_{}_{}.csv'
    features = pd.read_csv(path.format(feature_path, mode, rec, comb), 
                           index_col=0)

    X = features
    y = None
    if 'target' in features:
        X = features.drop('target', axis=1)
        y = np.log(features['target'])

    if stack_path:
        format_str = '{}/{}_predictions_stack_*_{}_{}.csv'
        stack_files = glob.glob(format_str.format(stack_path, mode, rec, comb))
        for file in stack_files:
            predictions = pd.read_csv(file, index_col=0)
            X = X.merge(predictions, left_index=True, right_index=True)

    return X, y


def fit_models_cross_validation(feature_path, output_path, stack_path):
    """Fit CatBoost using cross-validation on the features and other 
    classifiers' predictions if stack_path is provided. Perform this for
    every (recipe, process_combination) and print a table with results."""
    all_predictions = {}
    combinations = [itertools.product([rec], combinations_per_recipe[rec]) 
                    for rec in combinations_per_recipe]
    combinations = list(itertools.chain.from_iterable(combinations))
    for (rec, comb) in combinations:
        X_train, y_train = load_data('train', rec, comb, feature_path, 
                                     stack_path)
        prediction_df = fit_cat_cv(X_train, y_train, rec, comb, 
                                   output_path=output_path)
        all_predictions[(rec, comb)] = prediction_df

    # Let's remove the (15, 15) processes from (9, 8)
    proc_15_15 = all_predictions[(15, 15)].index
    proc_9_8 = all_predictions[(9, 8)].index
    proc_9_8 = list(set(proc_9_8) - set(proc_15_15))
    all_predictions[(9, 8)] = all_predictions[(9, 8)].loc[proc_9_8]

    table_data = [['Recipe', 'Process Combination', 'Weight', 'MAPE']]
    total_mape = 0
    for (rec, comb) in combinations:
        path = '{}/train_features_{}_{}.csv'
        train_features = pd.read_csv(path.format(feature_path, rec, comb), 
                                     index_col=0)
        y = train_features['target']
        agg_predictions = []
        preds = all_predictions[(rec, comb)].reindex(y.index)
        agg_predictions.append(preds)
        mape = custom_mape(preds['prediction'], y)
        weighted_mape = weights[(rec, comb)] * mape
        total_mape += weighted_mape
        table_data.append([str(rec), str(comb), str(weights[(rec, comb)]), 
                           str(mape)])

    result_table = AsciiTable(table_data, 'MAPE per model')
    print(result_table.table)
    print('TOTAL MAPE = {}'.format(total_mape))


def fit_models_submission(feature_path, output_path, stack_path):
    """Generate a submission by fitting CatBoost  on the features and other 
    classifiers' predictions if stack_path is provided. Perform this for
    every (recipe, process_combination)."""
    submission = []
    for rec in combinations_per_recipe:
        for comb in combinations_per_recipe[rec]:
            log_str = 'Generating predictions for recipe {} & combination {}'
            logging.info(log_str.format(rec, comb))
            X_train, y_train = load_data('train', rec, comb, feature_path, 
                                         stack_path)
            X_test, _ = load_data('test', rec, comb, feature_path, 
                                  stack_path)

            path = '{}/{}_{}_{}'.format(output_path.rstrip('/'), rec, comb, 
                                        'submission')
            predictions = np.exp(fit_cat(X_train, y_train, X_test, 
                                         output_path=path))
            prediction_df = pd.DataFrame(np.reshape(predictions, (-1, 1)), 
                                         index=X_test.index, 
                                         columns=['prediction'])
            submission.append(prediction_df)

    today = datetime.datetime.now()
    submission_df = pd.concat(submission).sort_index().reset_index(drop=False)
    submission_df.columns = ['process_id', 'final_rinse_total_turbidity_liter']
    format_str = '{}/submission_{:02d}{:02d}.csv'
    submission_df.to_csv(format_str.format(output_path, today.month, 
                                           today.day), index=False)


@click.command()
@click.option('--cross_validation/--submission', default=False)
@click.argument('feature_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
@click.argument('stack_path', required=False)
def main(cross_validation, feature_path, output_path, stack_path=None):
    if cross_validation:
        fit_models_cross_validation(feature_path, output_path, stack_path)
    else:
        fit_models_submission(feature_path, output_path, stack_path)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
