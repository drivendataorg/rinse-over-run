#  Sustainable Industry: Rinse Over Run 15th Place Solution

This is an outline of how to reproduce the 15th place solution for the [Sustainable Industry: Rinse Over Run](https://www.drivendata.org/competitions/56/predict-cleaning-time-series/) competition.

## Contents
01-generate-features.py : code to process train and test data and generate features used in the final model  
02-fit-predict.py : code to estimate the model and make predictions on the test set  
selected_features.pickle : pickle file with a dictionary of lists of features used in the final model  

## Hardware and software
The following was used to create the original solution:  
n1-standard-8 (8 vCPUs, 30 GB memory) on Google Cloud Platform (although the solution should replicate with less resources)
Debian GNU/Linux 9.7  
Python 3.7.1

## Package requirements  
pandas==0.23.4  
numpy==1.15.4  
sklearn==0.20.1  
tqdm==4.28.1  
lightgbm==2.2.2  

## Steps to reproduce the solution
1. Create folders for the provided data, temporary files and the final output:
```
mkdir data tmp output
```
2. Download the data from [drivendata.org](https://www.drivendata.org/competitions/56/predict-cleaning-time-series/data/) to `data/`
3. Run:  
```
python 01-generate-features.py
python 02-predict.py
```

The final submission file is `output/sub.csv`
