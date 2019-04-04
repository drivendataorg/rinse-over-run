drivendata-cleaning-time
==============================

Drivendata Contest, Sustainable Industry: Rinse Over Run


How to run scripts
------------

    make create_environment create_dirs
    workon drivendata-cleaning-time-sol  # or conda activate drivendata-cleaning-time
    # download raw data into data/raw
    make requirements
    make data
    make gen_submission

Expected raw data
------------

Following files should be placed in data/raw

    recipe_metadata.csv
    submission_format.csv
    test_values.zip
    train_labels.csv
    train_values.zip

Hardware
--------

I was using MacBook Pro 2.7Ghz, 8GB RAM, OSX 10.14.4.
The solution is deterministic on OSX and recreate solution 100% identical to the
last submission. 
I've not tested it on Linux, but due to implementation differences in Keras,
there might be some minor differences.

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
