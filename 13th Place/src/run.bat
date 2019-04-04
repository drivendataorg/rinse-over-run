REM 1. Reshape training and test values
python reshape.py --raw_data_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\train_values.csv --input_labels_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\train_labels.csv --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp

python reshape.py --raw_data_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\test_values.csv --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp
REM 2. Tune training to find optimum number of estimators for each model
python tune.py --reshaped_training_data_path E:\Sustainable-Industry-Rinse-Over-Run\temp\train_values-reshaped-phases-combined.csv --recipe_metadata_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\recipe_metadata.csv --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp
REM 3. Train all models using reshaped training data
python train.py --reshaped_training_data_path E:\Sustainable-Industry-Rinse-Over-Run\temp\train_values-reshaped-phases-combined.csv --recipe_metadata_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\recipe_metadata.csv --tunning_info_path E:\Sustainable-Industry-Rinse-Over-Run\temp\tunning_info.json --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp
REM 4. Predict turbidity for reshaped test data
python predict.py --reshaped_test_data_path E:\Sustainable-Industry-Rinse-Over-Run\temp\test_values-reshaped-phases-combined.csv  --recipe_metadata_path E:\Sustainable-Industry-Rinse-Over-Run\downloads\recipe_metadata.csv --models_directory E:\Sustainable-Industry-Rinse-Over-Run\temp	 --target_squared_inverse_sums_path E:\Sustainable-Industry-Rinse-Over-Run\temp\target_squared_inverse_sums.json --output_directory E:\Sustainable-Industry-Rinse-Over-Run\temp	