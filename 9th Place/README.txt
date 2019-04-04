README to build model and make predictions for Driven Data Rinse over Run competition

By: mlearn
Date: March 2019

Version: 0acd12b211fa503483d3567761b136cfc95764fe31fc9cfd3844795f260d5aed

The R script create_predictions.R can be directly called.  All competition data must be in ../data/ and unpacked to .csv files.

It has the following dependencies:

- R.  I used version 3.3.3 but I assume any recent version of R will do.  My "Rscript" is installed /usr/local/bin - if yours is elsewhere you may need to edit the top line of create_predictions.R to point to the correct location.

- vtreat and data.table which can both be installed from CRAN

- lightgbm which can be installed from https://github.com/Microsoft/LightGBM/tree/master/R-package. Installation is easy on Linux and only slightly harder on a Mac.  

Don't worry about lightgbm spewing out loads of warnings about "No further splits".  I would recommend running the script with the output redirected away from the console.

The script requires 8GB of RAM and runs in 45 minutes on a c5.2xlarge AWS instance.

lightgbm seems to be very sensitive to exact version/platform so a precisely matching result may not be achieved.  I can forward my fitted model if wanted.