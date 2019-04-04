# Rinse_Over_Run

## Rinse Over Run:
Forecasting turbidity in F&B industrial equipment.     
<https://www.drivendata.org/competitions/56/predict-cleaning-time-series/>
## Authors:
The BI Sharpes: Pat Walsh & David Belton   
March 2019  
[Pat's Github](https://github.com/pat42w) 

## Generating submission:
 1. Copy the contest data to `/data` directory.
This data should be available at:
<https://www.drivendata.org/competitions/56/predict-cleaning-time-series/page/125/#datasets>

 2. Ensure the requirements in `requirements.txt` are satisfied.
     - I'm running this on Windows but this will work in python 3.7.1 with the requirements met.

 3. Getting Predictions:
	 - navigate to `Rinse_Over_Run` base folder
	- Run following python script: `Rinse_Over_Run.py`
 4. The recreated final submission is stored in `data/test_predictions.csv`

## Details about generating predictions:
When the script is running, you should see:
>(base) C:\Users`\YourNameHere\`Documents\GitHub\Rinse_Over_Run>`python Rinse_Over_Run.py`     
Begining data load this may take a few minutes   
train_values.csv uploaded  
recipe_metadata.csv uploaded  
train_labels.csv uploaded  
test_values.csv uploaded  
Launching  
Training Data prep successful  
Test Data prep successful  
Object Flow1 successful  
Clustering Flow2 successful  
Rinse Over Run successful: prediction can be found at data/test_predictions.csv  
...It's been done.  

Heres an explanation of these progress updates:  

Begining data load this may take a few minutes.     
  > - confirming required libraries have been imported and functions have been defined correctly

`train_values.csv` uploaded 
> - train_values.csv found and read

`recipe_metadata.csv` uploaded 
> - recipe_metadata.csv found and read

`train_labels.csv` uploaded 
>- train_labels.csv found and read

`test_values.csv` uploaded 
> - test_values.csv found and read

Launching 
>- other variables defined and the model workflow is being launched

Training Data prep successful 
> - Prepped the training data for the Flows

Test Data prep successful 
> - Prepped the test data for the Flows

Object Flow1 successful 
> - Object Flow1 executed succesfully

Clustering Flow2 successful 
>- Cluster Flow2 executed succesfully

Rinse Over Run successful: prediction can be found at `data/test_predictions.csv`  
...It's been done.  
 > - Verifying completion  

 ...It's been done.   
 Referencing a **[classic Simpsons quote](https://www.youtube.com/watch?v=eb1viD56zkM)**.
   
It takes about **10 minutes** to for the script to execute and recreate the solution, the most time consuming part is uploading the csv's as they are quite large at 2.13GB.