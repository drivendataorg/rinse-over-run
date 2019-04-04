## RINSE OVER RUN

**Approach Explaination**

We predict turbidity of a process using multiple xgboost models, where each model predict turbidity for a single phase or combinations of phases. This approach is building optimized models based on testing data structure and evaluation metric. 

### First : how did we build models that are optimized based on testing data structure?
In test data, We are given data for certain phase(s) of process, so we collect statistics for each phase in the process, then we feed this statistics to xgboost model to train or predict. To predict turbidity of a process, we get the median for all predictions for different phases. For example, let us have a process of two phases(pre_rinse, Caustic), then we collect statistics for pre_rinse phase data, caustic phase data, and statistics for both phases combined together, so, We have three rows corresponding to this process, then when We predict turbidity for this process, we predict turbidity for each row of this process, then We get the median for these values as the final turbidity of the whole processs.

At beginning, we built a single model to predict turbidity for all phases,and we got a good accuracy, but when we built a separate model to predict turbidity of each phase, or phases combinations, we got a higher accuracy because each model is now optimized for data of this phase or this combination of phases only. So, we have `15` different xgboost models. For more explaination check below section.

The process has up to five phases, but we only collect statistics for up to four phases, because the final rinse phase represents the output. Let us represent process phases as four digits binary number, where each digit represents existence of certain phase in the process, where the lowest bit represents pre_rinse phase, and highest bit represents acid phase. For example, if we have a process with code = `1011`, it means that process has pre_rinse, caustic ,and acid phases, but no intermediate rinse phase. So, we should have 16 different combination for process phases, but `0000` is not available as there is no process without any phases at all, so we have only 15 different combination of phases, and in consequence of this, we have 15 different light models, where each of them represents combination of phases in process.

To get more familiar with the approach, let us analyze data for previously mentioned process code= `1011`, when we preprocess/reshape data for this process, we will have 7 rows representing this process
1. a row represents statistics for pre_rinse phase
2. a row represents statistics for caustic phase
3. a row represents statistics for acid phase
4. a row represents statistics for pre_rinse+caustic phases combined
5. a row represents statistics for pre_rinse+acid phases combined
6. a row represents statistics for caustic+acid phases combined
7. a row represents statistics for pre_rinse+caustic+acid phases combined

Then each row is fed to corresponding model for training and prediction, then we get median of all predictions for all rows predictions.
Now we have models optimized based on test data structure, 

###  Second: How did we optimize our models based on evaluation metric?

XGBoost models can`t optimize MAPE metric directly, so we needed to use weighted version of target variable to be optimized. Check this video for more details about optimizing XGBoost model for MAPE metric
https://youtu.be/JaG-nFlU-jo?list=PLpQWTe-45nxL3bhyAJMEs90KF_gZmuqtm&t=310

So, we transform our target variable, `final_rinse_total_turbidity_liter`, to a weighted value that can be optimized using XGBoost model, and after prediction we re-tranform it back to its actual value.

Another optimization is related to using variant version of MAPE, The weighted version of target variable set higher weights for lower values, but we need to have all values below threshold(290000) having the same weight. So, we built another `15` light models corresponding each phases combinations for training and predicting for processes with `final_rinse_total_turbidity_liter < threshold` without weighting.

Because MAPE set higher penalaties for lower values, the final optimization is that we scaled down the predicted value, and this led to further improvement in score.


**Software Requirements**
Before running any command, you need to do 
- Install python. I am using version `3.6.1`
- Open CMD or shell and navigate to `src` directory
- Install required packages -> `pip install -r requirements.txt`


**Code Explanation and Execution**
- Step1: `reshape.py` Reshape the raw data to proper input data formats. We do this transformation for both testing and training data based. It does the following sub-steps
    1. Load train or test values
    2. Drop columns that are not used in modeling: [`row_id`, `object_id`, `timestamp`, `target_time_period`]
    3. Remove final rinse data for all processes in training data
    4. Load train labels, and merge them with train values based on `process_id`
    5. Group data by `process_id` and then group each process data by `phase`
    6. Get statistics,[`mean`, `std`, `median`, `min`, `max`, sum of turbidity during this phase, ...], for each phase
    7. Save reshaped data to csv file to be used in training or prediction

    You can use command line to reshape data for both training and testing values. Open CMD, and navigate to current working directory issue the following commands
    - For reshaping training data:

    `python reshape.py --raw_data_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\train_values.csv --input_labels_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\train_labels.csv --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp`
    
    - For reshaping test data
    
    `python reshape.py --raw_data_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\test_values.csv --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp`

    where
    - `raw_data_path` is the raw data provided in challenge data
    - `input_labels_path` is path of file containing labels of training data. It's passed only in case of reshaping training data
    - `output_directory` is where reshaped data will be saved

- Step2: `tune.py`  We use kFold cross validation to find optimum number of estimators for training each model. You can use CMD to tune the model
    - `python tune.py --reshaped_training_data_path E:\Sustainable-Industry-Rinse-Over-Run\temp\train_values-reshaped-phases-combined.csv --recipe_metadata_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\recipe_metadata.csv --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp`

    where
    - `reshaped_training_data_path` is path of file generated for training data from previous step
    - `recipe_metadata_path` file containing process recipe metadata
    - `output_directory` is where tunning information of models are saved

- Step3: `train.py` We train our models against reshped training data using tuned number of estimators for each model. You can use CMD to tune the model
    - `python train.py --reshaped_training_data_path E:\Sustainable-Industry-Rinse-Over-Run\temp\train_values-reshaped-phases-combined.csv --recipe_metadata_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\recipe_metadata.csv --tunning_info_path E:\Sustainable-Industry-Rinse-Over-Run\temp\tunning_info.json --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp`

    where
    - `reshaped_training_data_path` is path of file generated for training data from step1
    - `recipe_metadata_path` file containing process recipe metadata
    - `tunning_info_path` file tunning information generated from step2
    - `output_directory` is where models information of models will be saved
- Step4: `predict.py` Load reshaped data and predict turbidity for each row using corresponding model, then get median for all rows of each process as the predicted turbidity for this process. You can use CMD to tune the model
    - `python predict.py --reshaped_test_data_path E:\Sustainable-Industry-Rinse-Over-Run\temp\test_values-reshaped-phases-combined.csv  --recipe_metadata_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\recipe_metadata.csv --models_directory E:\Sustainable-Industry-Rinse-Over-Run\temp	 --target_squared_inverse_sums_path E:\Sustainable-Industry-Rinse-Over-Run\temp\target_squared_inverse_sums.json --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp	`

    where
    - `reshaped_test_data_path` is path of file generated for testing data from step1
    - `recipe_metadata_path` file containing process recipe metadata
    - `target_squared_inverse_sums_path` file contains some information about to convert weighted predicted value to its actual value.
    - `models_directory` is where models information of models are saved
    - `output_directory` is where prediction for testing data will be saved

You can execute `run.bat` from `src` directory to run all previous steps in single command, but make sure to change paths to valid paths where raw data exists in your machine. Also, If you are running linux OS, change paths to be unix style.
The running time for `run.bat` on my machine(Intel i5 CPU with 2 physical cores @2.67GHz , 6144MB RAM) is about `90` Minutes


