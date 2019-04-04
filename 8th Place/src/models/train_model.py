import numpy as np
import lightgbm as lgb
import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV

from src.models.metrics import scoring, print_csvresult
from src.models.model_selection import ProcessSplit

warnings.filterwarnings("ignore")
tf = pd.read_csv("data/processed/tf.csv")
ef = pd.read_csv("data/processed/ef.csv")
submission_format = pd.read_csv("data/raw/submission_format.csv", index_col=0)

Regressor = lgb.LGBMRegressor
regressor = Regressor()

training_params = {
    "boosting_type": ["dart"],
    "objective": ["regression_l1"],
    "learning_rate": [0.02],
    "n_estimators": [2000],
    "max_depth": [12],
    "num_leaves": [29],
}


# Define target
target = "final_rinse_total_turbidity_liter"

# Exclude some columns
excludes = {
    target,
    "process_id",
    "weight",
    "phase_id_sim",
    "phase_id",
    "return_temperature|sum",
    # "pipeline",
    # "metadata",
    # "supply_flow|sum",
    # "return_flow|sum",
    # "supply_pressure|sum",
}
ts_to_exclude = [
    # "supply_flow",
    # "supply_pressure",
    # "return_temperature",
    # "return_turbidity",
    # "return_flow",
    "return_conductivity",
    "tank_level_pre_rinse",
    "tank_level_caustic",
    "tank_level_acid",
    "tank_level_clean_water",
    "tank_temperature_pre_rinse",
    "tank_temperature_caustic",
    "tank_temperature_acid",
    "tank_concentration_caustic",
    "tank_concentration_acid",
]
event_to_exclude = [
    # "return_drain",
    # "supply_caustic",
    # "supply_pre_rinse",
    "supply_pump",
    "return_caustic",
    "return_acid",
    "return_recovery_water",
    "supply_acid",
    "supply_clean_water",
]

for key in ts_to_exclude + event_to_exclude:
    for col in tf.columns:
        if "|" in col and col.split("|")[0] == key:
            excludes.add(col)
for col in tf.columns:
    if "|" in col and col.split("|")[1] in (
        "diff_avg",
        "diff_abs_avg",
        "changes_count",
    ):
        excludes.add(col)
for col in tf.columns:
    if "|" in col and col.split("|")[1].startswith("percentile"):
        excludes.add(col)
columns = list(set(tf.columns) - excludes)

# Define categorical features
categorical = ["object_id", "metadata", "pipeline", "phase_id"]
categorical = list(set(categorical) - excludes)

# Target reweighting
weights = tf["weight"].copy()
thr1 = 1002394.1870403971
thr2 = 8158920.013280024
weights.loc[tf[target] >= thr2] *= 1e-5
weights.loc[(tf[target] >= thr1) & (tf[target] < thr2)] *= 0.5280736958854724

fit_params = {"categorical_feature": categorical, "sample_weight": weights}
cv = GridSearchCV(
    regressor,
    training_params,
    fit_params=fit_params,
    cv=ProcessSplit(seed=0, k=5, tf=tf),
    scoring=scoring,
    verbose=1,
    n_jobs=3,
    refit=False,
)
print(f"Learning on a {len(tf)}x{len(columns)} matrix")

cv.fit(tf.loc[:, columns], np.ravel(tf.loc[:, target]))

# # of Estimators is increased by 10% for refitting as suggested by laurae (https://sites.google.com/view/lauraepp/parameters?authuser=0)
regressor = Regressor(
    **{
        **cv.best_params_,
        "n_estimators": int(cv.best_params_.get("n_estimators", 100) * 1.1),
    }
)
regressor.fit(tf.loc[:, columns], np.ravel(tf.loc[:, target]), **fit_params)
cv.best_estimator_ = regressor
print(f"Best param for scoring={scoring}: {-cv.best_score_}, {cv.best_params_}")

print_csvresult(cv, training_params, columns=columns)

# Predict submission
my_submission = pd.DataFrame(
    data=regressor.predict(ef[columns]).tolist(),
    columns=submission_format.columns,
    index=submission_format.index,
)
my_submission.to_csv("data/processed/submission.csv")
