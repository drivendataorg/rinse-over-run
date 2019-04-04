# Schneider data challenge

[Sustainable Industry: Rinse Over Run](https://www.drivendata.org/competitions/56/)
data competition.

## Introduction

We settled on a gradient boosted tree to solve the problem. We chose [CatBoost](https://catboost.ai/)'s implementation of the algorithm, as we have found it to be provide generally better results than XGBoost.

## How to run the model

### Prepare the data

Put raw data files in `data/raw`:
* `recipe_metadata.csv`
* `submission_format.csv`
* `test_values.zip`
* `train_labels.csv`
* `train_values.zip`

### Pull the Docker image

Install [Docker](https://www.docker.com/), then pull our Docker image: `docker pull contiamo/schneider`

### Execute the three notebooks one after the other:
1. `data_processing.ipynb` (train/test split and truncation of phases)
1. `feature_engineering.ipynb` (calculation of timeseries features)
1. `catboost/best_model.ipynb` (training the model)

Specifically:

```bash
docker run --rm -e CHOWN_HOME=yes -v "$PWD":/home/jovyan/work contiamo/schneider papermill /home/jovyan/work/notebooks/data_processing.ipynb /home/jovyan/work/notebooks/data_processing.output.ipynb
docker run --rm -e CHOWN_HOME=yes -v "$PWD":/home/jovyan/work contiamo/schneider papermill /home/jovyan/work/notebooks/feature_engineering.ipynb /home/jovyan/work/notebooks/feature_engineering.output.ipynb
docker run --rm -e CHOWN_HOME=yes -v "$PWD":/home/jovyan/work contiamo/schneider papermill /home/jovyan/work/notebooks/catboost/best_model.ipynb /home/jovyan/work/notebooks/catboost/best_model.output.ipynb
```

The resulting submission will be in `data/`.

## Interactive mode

In order to run the notebooks interactively:

```bash
docker run --rm -p 127.0.0.1:8888:8888 -e CHOWN_HOME=yes -v "$PWD":/home/jovyan/work contiamo/schneider jupyter lab --NotebookApp.token=''
```

Then open your browser at this address: http://localhost:8888. If the port is already in use, change it in the `-p` flag above.

This can be particularly interesting for the last, modeling notebook.

**Note**: In order to visualize progression during model training, the notebook has to viewed in the "classic" view: `Help > Launch Classic Notebook`.


## Annex: Description of processed data

We split the original data into the following datasets, available as parquet files:
* `train_ts_truncated`: 70% of original train data, with final_rinse removed, and truncated to match phase distribution in hidden test set;
* `test_ts_truncated`: 30% of original train data, with final_rinse removed, and truncated to match phase distribution in hidden test set;
* `train_target`: target value for processes in `train_ts_truncated`;
* `test_target`: target value for processes in `test_ts_truncated`.
