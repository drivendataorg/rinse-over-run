# Sustainable Industry: Rinse Over Run

This repository contains the **10th** place solution for the competition  
https://www.drivendata.org/competitions/56/predict-cleaning-time-series/


## **How to prepare:**

### 1. Install anaconda


- Prepare the system

`sudo apt-get update`

`sudo apt upgrade`
- Choose the _existing_ folder to install the anaconda, here `/home/ubuntu/anaconda_installation` is just an example

`anaconda_installation_folder="/home/ubuntu/anaconda_installation"` 


`cd ${anaconda_installation_folder}`

`sudo curl -O https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh`

`sudo bash Anaconda3-5.2.0-Linux-x86_64.sh`


- Now anaconda installer will ask you some questions.
Correct **answers** are:

`In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
-------->` **ENTER**

Then you'll see a lot of text, scroll it pressing **ENTER**

`Do you accept the license terms? [yes|no]
-------->` **Yes**

`Press ENTER to confirm the location
 Press CTRL-C to abort the installation
 Or specify a different location below
-------->` **the name of anaconda_installation_folder + /anaconda3 (for example /home/ubuntu/anaconda_installation/anaconda3 in this script)**

`Do you wish the installer to prepend the Anaconda3 install location
to PATH in your /home/natasha/.bashrc ? [yes|no]
-------->` **Yes**

`Do you wish to proceed with the installation of Microsoft VSCode? [yes|no]
-------->` **No**

### 2. Prepare anaconda for our task

- Add channels to use for packages installation

`cd ${anaconda_installation_folder}/anaconda3/bin`

`sudo ./conda config --append channels conda-forge`

`sudo ./conda config --append channels pytorch`

- Grant rights to the user ubuntu for installing packages 

`sudo chown -R ubuntu:ubuntu ${anaconda_installation_folder}/anaconda3`

### 3. Create and activate the environment

`sudo ./conda create --name rinse`

`source activate rinse`

### 4. Install packages
- `conda update -n base conda`
- `conda install tqdm`
- `conda install pandas`
- If you have a GPU install a GPU version of Pytorch 
with the correspondent version of CUDA 
(here 9.0 is an example, to get your version of CUDA type `nvcc --version`)

`conda install pytorch torchvision cudatoolkit=9.0 -c pytorch`

otherwise, if you only have a CPU install a CPU version of Pytorch

`conda install pytorch-cpu torchvision-cpu -c pytorch`

- `pip install jstyleson`


After you've installed all packages you should use the python from your fresh virtual environment
to run any  scripts

`${anaconda_installation_folder}/anaconda3/envs/rinse/bin/python3.7`

### 5. Adjust the config file

Under the root of this project, there is a file `config.json`.
You should put there 
- the information of GPU availability

`"device": "cuda:0"` if you have a GPU or 

`"device": "cpu"` if you have the CPU only
- the information about the data location on your machine

`"data_dir_control": "/media/natasha/Data1/rinse_data"` path to the folder 
containing **test_values.csv**

`"data_dir_train": "/media/natasha/Data1/rinse_data"` path to the folder 
containing **train_values.csv**

`"temp_folder": "/media/natasha/Data1/temp/all_outputs_rinse/"` path to the _existing_ folder 
for storing the intermediate results (make sure that your user has rights to write to this folder)

## **How to run:**

##### First of all go to the root folder of the project.

##### There are 2 ways of using this code:
###### 1. Just do the ***inference*** with the pre-trained model 
Should closely reproduce the result for the competition. 

To use the code for inference you should change the parameter in the `config.json` file:

`"only_inference": true`


###### 2. ***Train*** the new model and do the inference
(can have slightly different weights, so won't exactly reproduce the results for the competition)
and then use the newly trained model to perform the inference. 
This approach _allows you to use the new data_ for training and possibly improve the final results.

To use the code for train and inference you should change the parameter in the `config.json` file:

`"only_inference": false`

Also, depending on the memory available on your computer 
you can change the batch size parameter in the `config.json` file:
 
`"batch_size_train": 64` is suitable for 6Gb of GPU memory and 8 Gb CPU memory, increase/decrease according to your resources 

##### Then in both modes you should do the following:

- Run the `train.py` script with the python from our virtual environment and our `config.json` file as a parameter:

`${anaconda_installation_folder}/anaconda3/envs/rinse/bin/python3.7 /home/natasha/PycharmProjects/rinse/train.py -c config.json`

This script will produce the new file `submission_test.csv` under the root of the project. 

- The above file `submission_test.csv` requires a postprocessing, now run the `postprocessing_after_tta.py` script with the python from our virtual environment:

`${anaconda_installation_folder}/anaconda3/envs/rinse/bin/python3.7 /home/natasha/PycharmProjects/rinse/postprocessing_after_tta.py`

This script will produce the new file `submission_test_tta.csv` under the root of the project. 
**This file contains the final result.** 

In the case of the _inference_ mode it should be close to the results in my best submission file 
`submission_test_tta_0.2837.csv` which is located under the root of the project.
The result is not exactly the same due to the stochasticity in test time augmentation.

In the case of the _train_ mode, the result also can be slightly different 
because of stochasticity in the training procedure. 
The newly trained model weights can be found under the folder `saved/Rinse`. 

If you want to use these new weights further you can store them in any folder you like 
and use them in the inference mode. To do so go to `train.py` and change the inference-only 
weights folder from `saved/submission_best` to the folder with your new weights:

`inference_only_weights_folder = 'saved/submission_best'`
