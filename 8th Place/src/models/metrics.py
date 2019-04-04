import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


def mean_absolute_percentage_error(y_true, y_pred):
    """
        Evaluation metric
    """
    n = len(y_true)
    thr = 290000
    return (
        np.sum(np.abs(y_true - y_pred) / np.maximum(np.ones(n) * thr, np.abs(y_true)))
        / n
    )


scoring = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


def print_csvresult(cv, params, columns, n_features=40, **kwargs):
    metrics = [
        "mean_train_score",
        "mean_test_score",
        "mean_fit_time",
        "mean_score_time",
    ]
    cvresults = pd.DataFrame(cv.cv_results_)[["params"] + metrics]
    for key in params:
        cvresults[key] = cvresults["params"].apply(lambda x: x[key])
    for key, value in kwargs.items():
        cvresults[key] = value
    cvresults = cvresults[list(params.keys()) + list(kwargs.keys()) + metrics]
    cvresults = cvresults.sort_values("mean_test_score", ascending=False)
    print(cvresults)
    for key, values in params.items():
        if len(values) > 1:
            print(cvresults.groupby(key)["mean_test_score"].mean().sort_values())

    print(
        pd.Series(cv.best_estimator_.feature_importances_, index=columns)
        .sort_values(ascending=False)[:n_features]
        .to_frame("feature")
    )

    features_importance = pd.Series(
        cv.best_estimator_.feature_importances_, index=columns, name="count"
    ).reset_index()
    features_importance["original_feature"] = features_importance["index"].apply(
        lambda x: x[0] if isinstance(x, tuple) else x.split("|")[0]
    )
    features_importance["funcname"] = features_importance["index"].apply(
        lambda x: x[1].split("_")[2]
        if isinstance(x, tuple) and x[1].startswith("last")
        else -1
    )
    print(features_importance.groupby("original_feature").sum()["count"].sort_values())
    return cvresults
