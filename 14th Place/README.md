# Efficiency Project

This report contains the analysis and final model used to submit the best private score in the leader board on the Efficiency competition hosted by **DrivenData**.

## Overview

The goal of the competition was to predict the **final rinse total turbidity liter** (target) for a cleaning process of a particular object. The final model consisted on 5 steps.

1. Pre-processing and feature engineering: Each process was divided in 5 different phases and it  recorded different measurements of the cleaning status every 2 seconds. We summarize each of the measurement by phase, obtaining 1 or 2 statistics of each measure. Additionally to this,we calculated the average of the target by object excluding outliers

2. Subset the data: Since each process had recorded different phases we subset the test data according of the observed phases in 6 different groups and we created 6 different training set for each group.

3. Train independent random forests: For each group we trained and tune independently 6 random forests, most of the variables kept in each forest were similar among each other.

4. Predict: For each process in the test we used the random forest that corresponded to the observed phases of the process.


## Dependencies


This model ran in R version 3.4.1 and used the following packages:

* tidyverse  (1.1.1)
* magrittr   (1.5)
* stringr    (1.2.0)
* lubridate  (1.6.0)
* ranger     (0.8.0)
* ProjectTemplate (0.8)

## Reproduce final submission

To reproduce final submission you should save the data inside the data folder using the original file names given by the organizers and execute the following commands inside R 

```
#This will load the data and libraries required
library(ProjectTemplate)
load.project()
```

The file `src/Final_submission_model.R` contains the script that produces the exact submission that created the best private score.

There are only 175 lines of code which highlight the simplicity of the model, the object `Submission` contains the final predictions. 

