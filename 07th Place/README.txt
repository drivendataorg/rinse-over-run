The model is very similar to the example model http://drivendata.co/blog/rinse-over-run-benchmark. The total model consists of 4 lightgbm models which are located in the file lightgbm_Allmendinger.py.
All process steps, such as reading the data, preparing the data, fitting the model and generating the predictions (submit file) are in this file.
As a development environment, I use Windows 10, Python 2.7.To run the model i use the anaconda package spyder.
To run the script (lightgbm_Allmendinger.py), the libraries 
1.	Pathlib
2.	Numpy
3.	Pandas
4.	Lightgbm (conda install -c conda-forge lightgbm)
must be installed. 
It is important that for lightgbm the latest version is installed (training with objective 'mape' should be run without error)

The raw data must be in the subdirectory data.
If you run excute the script (lightgbm_Allmendinger.py) the raw data are read in, prepared, the model is fitted; the forecasts for the test data are generated and written to the disk in the file with name 'submission_GMBlight_27.csv'.
