# Rinse Over Run

The code for the winning solution of the [Rinse Over Run competition](https://www.drivendata.org/competitions/56/predict-cleaning-time-series/), hosted by DrivenData.

## Requirements and install

- Quite a significant amount of RAM memory is required (~10GB). It should be noted that the code is not optimized, and therefore the memory requirement can easily be reduced by adapting some parts of the code.
- Running all steps will take roughly 30-40 hours. Some easy optimizations to reduce this time include running CatBoost on GPU, removing the Random Forest Classifier with 250 trees from the stack, removing some of the most computationally expensive features from tsfresh, and disabling some insignificant models ((3, 2), (15, 2), (15, 6), (15, 14)).
- We added a `requirements.txt` with all dependencies. Just run `pip install -r requirements.txt`.

## Building the features

`build_features.py [OPTIONS] TRAIN_PATH TEST_PATH LABEL_PATH RECIPE_PATH OUTPUT_PATH`

**Example:** `python3 src/features/build_features.py data/raw/train_values.csv data/raw/test_values.csv data/raw/train_labels.csv data/raw/recipe_metadata.csv data/features/`

**Output:** Files will be created in `data/features`

## Generating out-of-sample predictions for stacking

`stacking.py [OPTIONS] FEATURE_PATH OUTPUT_PATH`

**Example:** `python3 src/models/stacking.py data/features/ data/predictions/`

**Output:** Files will be created in `data/predictions`

## Evaluate models with cross-validation

`gradient_boosting.py --cross-validation FEATURE_PATH OUTPUT_PATH [STACK_PATH]`

**Example:** `python3 src/models/gradient_boosting.py --cross_validation data/features/ output/`

**Output:** Shapley plots in `output` and an ASCII Table with MAPES to stdout.

### Results without stacking

```
+MAPE per model----------------+--------+---------------------+
| Recipe | Process Combination | Weight | MAPE                |
+--------+---------------------+--------+---------------------+
| 3      | 1                   | 0.0219 | 0.38150749103694254 |
| 3      | 2                   | 0.0064 | 0.30265204730067097 |
| 3      | 3                   | 0.1695 | 0.3036406446617803  |
| 9      | 8                   | 0.0411 | 0.261495246538695   |
| 15     | 1                   | 0.0765 | 0.3113903339369102  |
| 15     | 2                   | 0.0013 | 0.28706253186845304 |
| 15     | 3                   | 0.2289 | 0.28177810261072755 |
| 15     | 6                   | 0.0007 | 0.27781473057860173 |
| 15     | 7                   | 0.2258 | 0.2799518406911836  |
| 15     | 14                  | 0.0017 | 0.2539413686237908  |
| 15     | 15                  | 0.2262 | 0.2523300341218434  |
+--------+---------------------+--------+---------------------+
TOTAL MAPE = 0.2821164305690393
```

### Results with stacking

```
+MAPE per model----------------+--------+---------------------+
| Recipe | Process Combination | Weight | MAPE                |
+--------+---------------------+--------+---------------------+
| 3      | 1                   | 0.0219 | 0.35104325780935636 |
| 3      | 2                   | 0.0064 | 0.2957717269706243  |
| 3      | 3                   | 0.1695 | 0.28415849376612    |
| 9      | 8                   | 0.0411 | 0.25423865019668124 |
| 15     | 1                   | 0.0765 | 0.30243442762859934 |
| 15     | 2                   | 0.0013 | 0.2915724006451775  |
| 15     | 3                   | 0.2289 | 0.28541422151362134 |
| 15     | 6                   | 0.0007 | 0.2789651608133909  |
| 15     | 7                   | 0.2258 | 0.28167597813837186 |
| 15     | 14                  | 0.0017 | 0.2538038655453939  |
| 15     | 15                  | 0.2262 | 0.25084776091275696 |
+--------+---------------------+--------+---------------------+
TOTAL MAPE = 0.2780123943200789
```

## Create submission

`gradient_boosting.py --submission FEATURE_PATH OUTPUT_PATH [STACK_PATH]`

**Example:** `python3 src/models/gradient_boosting.py --submission data/features/ output/`

**Output:** A submission file in `output/`

## Project Organization

**Make sure your directory is structured as follows by creating the required directories (such as output and the different data directories).**

```
    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── features       <- The extracted features from the timeseries.
    │   ├── predictions    <- Predictions generated through stacking.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── output             <- Generated graphics/plots and submission files.
    │
    ├── notebooks          <- Jupyter notebooks exported from Google Colab
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and make predictions
    │   │   ├── stacking.py
    └── └── └── gradient_boosting.py
```

