1. I'm Guillaume Thomas, french, 32y old, husband and father of two kids. I've a master degree from Ecole Centrale Paris in Applied Mathematics and from Ecole Normale Supérieure in Maths, vision & learning. I've started my career in the ecommerce area working on topics like product clustering, product recommendation, click-rate prediction & real-time-bidding. I'm currently the CTO of InUse, an IOT platform dedicated to the industry area (especially food & beverage & energy). We're able to deliver several services: supervision, predictive maintenance, cip. I'm in charge of the product development and participate to customer projects which require advanced data analysis.

2. My approached consisted of:
    1. For each process and for each process step, consider a subprocess with data only up to this step for the training. Associate a Weight accordingly to the test distribution (0.1 for process ending at pre_rinse, 0.3 otherwise).
    2. Generate for each subprocess simple time-domain aggregations
    3. Focus on ensemble methods (especially lightgbm)
    4. Reduce weights of high targets which penalized the evaluation metrics

3. Here are the most impactful parts of my code:

###### Generate subprocess

The code below basically transforms the input features matrix of ~5k rows to a another feature matrix of ~17 rows with weights adapted to test set. Each process generates N subprocess, N being the count of run steps.

```python
class SimulationTransformer(BaseTransformer):
    def __init__(self, simulate):
        self.simulate = simulate

    def simulators(self):
        return pd.DataFrame(
            data=[
                [1234, 1, .1],
                [1234, 12, .3],
                [1234, 123, .3],
                [1234, 1234, .3],
                [234, 2, .333],
                [234, 23, .333],
                [234, 234, .333],
                [12, 1, .1],
                [12, 12, .9],
            ]
            if self.simulate
            else [],
            columns=["input_recipe", "observed_recipe", "weight"],
        )

    def set_simulate(self, simulate):
        self.simulate = simulate

    def transform(self, X):
        ret = X.reset_index().merge(self.simulators(), on="input_recipe", how="left")
        ret["weight"] = ret["weight"].fillna(1)
        idx = pd.isnull(ret["observed_recipe"])
        ret.loc[idx, "observed_recipe"] = ret.loc[idx, "input_recipe"]
        ret["observed_recipe"] = ret["observed_recipe"].astype(int)
        return ret
```

###### Target reweighting

In the following code, i modify the weights based on the target. As explained in the report, this trick is done to compensate the misalignment between the evaluation metric (mix of MAE & MAPE) and the objective function (MAE).

```python
weights = tf["weight"].copy()
thr1 = 1002394.1870403971
thr2 = 8158920.013280024
weights.loc[tf[target] >= thr2] *= 1e-5
weights.loc[(tf[target] >= thr1) & (tf[target] < thr2)] *= 0.5280736958854724
```

###### Hyper parameters bayesian optimization

In the following code, i've used the hyperopt library to perform hyper parameters optimization. In this example, we explore several parameters domain:
* thr1, thr2, tau1 & tau2: target reweighting parameters
* num_leaves, max_depth, min_data_per_group, cat_smooth, colsample_bytree: lightgbm hyperparameters

```python
def objective(params):
    regressor = Regressor(**base_params)
    weights = tf["weight"].copy()
    thr1 = params["thr1"]
    thr2 = params["thr2"]
    weights.loc[(tf[target] >= thr1) & (tf[target] < thr2)] *= params["tau1"]
    weights.loc[tf[target] >= thr2] *= params["tau2"]

    fit_params = {
        "categorical_feature": list(set(categorical) - excludes),
        "sample_weight": weights,
    }
    cv = GridSearchCV(
        regressor,
        param_grid={
            "num_leaves": [int(params["num_leaves"])],
            "max_depth": [int(params["max_depth"])],
            "lambda_l1": [params["lambda_l1"]],
            "min_data_per_group": [int(params["min_data_per_group"])],
            "cat_smooth": [params["cat_smooth"]],
            "colsample_bytree": [params["colsample_bytree"]],
        },
        fit_params=fit_params,
        cv=ProcessSplit(seed=0, k=5),
        scoring=scoring,
        verbose=0,
        n_jobs=3,
        refit=False,
    )

    cv.fit(tf.loc[:, columns], np.ravel(tf.loc[:, target]))
    print(f"Best score: {cv.best_score_}, {params}")
    return {"loss": -cv.best_score_, "params": params, "status": STATUS_OK}


space = {
    "num_leaves": hp.quniform("num_leaves", 20, 50, 2),
    "max_depth": hp.quniform("max_depth", 5, 15, 1),
    "thr1": hp.loguniform("thr1", np.log(7e5), np.log(3e6)),
    "thr2": hp.loguniform("thr2", np.log(3e6), np.log(2e7)),
    "tau1": hp.loguniform("tau1", np.log(1e-5), np.log(1)),
    "tau2": hp.loguniform("tau2", np.log(1e-5), np.log(1)),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    "lambda_l1": hp.loguniform("lambda_l1", np.log(1e-4), np.log(1e-1)),
    "min_data_per_group": hp.quniform("min_data_per_group", 5, 100, 5),
    "cat_smooth": hp.loguniform("cat_smooth", np.log(1e-2), np.log(10)),
}
trials = Trials()
best = fmin(
    fn=objective, space=space, algo=tpe.suggest, max_evals=500, trials=trials, verbose=3
)
```

4. Here are the other things tried:
    1. Other models (randomforest & xgboost)
    2. Other aggregations:
        1. N-th Percentiles (ex: 10th percentile). 
        2. Average on n-th last rows.
        3. Usage of [tsfresh package](https://github.com/blue-yonder/tsfresh) which compute many time series features. I focused on frequency-domain aggregations like `fft_aggregated` but it did not help.
    3. Models bagging: I tried an average of two regressors: one optimized for low target values and one for high target values.

5. I only used jupyter notebook for data exploration & models training.

6. Not really.

7. None

8. Most of the charts are in the final report.

9. What would interest me is to get closer to the integration in and industrial process. In my company, we work with industrial companies which sell services related to the CIP. They mostly work on conductivity and do not consider turbidity that much. I'd like to understand more the relationship between the turbidity prediction and the water savings: which decision an operator or a machine can make based upon this prediction? How much water would we be able to save?
