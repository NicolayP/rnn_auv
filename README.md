# Requirments:

uses torch 1.13 which requires cuda 11.6/11.7

install python requirments with the requiremnts.txt file

# usage:

There is one ros bag with the raw data stored in ""data/timestamp/run0.bag"

There is one cvs file with resampled data at 10Hz with different data representations. 

the file is stored in "./test_data_clean/csv/run0.csv".

The code to generate the csv file is clean_bag.py

The code with the integration of the ground truth velocity is 'debug.ipynb'.
