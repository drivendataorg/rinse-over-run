# rinse-over-run

## Organization

* Follows [drivendata datascience cookiecutter](https://drivendata.github.io/cookiecutter-data-science/)
* Directory organization follow scikit-learn organization. Examples:
    * models.metrics
    * models.model_selection
* Used [black](https://github.com/ambv/black) for automatic code formatting
* Code use python 3.6+. Check .python-version for the exact version i've used

## Install

I recommend using [pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).

Example:
```bash
$> pyenv virtualenv rinse
```

Otherwise, make sure you use python 3.6+.

Then, install the requirements.

```bash
$> python -m pip install -U pip setuptools wheel
$> python -m pip install -r requirements.txt
```

## Compute feature matrix

A first step consist of transforming the train & test values (which are time series) to agregations per process. To achieve this, run the following command:

```bash
$> mkdir -p data/raw/
$> mkdir -p data/processed/
$> python src/data/make_dataset.py data/raw/ data/processed/
```

It'll generate two files:
* Train features: data/processed/tf.csv
* Test features: data/processed/ef.csv

## Build the model & make the prediction

To build the model, run the following command:

```bash
$> python src/models/train_model.py
```

Trained model is not saved but it'll generate `data/processed/submission.csv`.

## Submission

Two markdown files a