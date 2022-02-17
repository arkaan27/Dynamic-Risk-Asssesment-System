"""
Name: diagnostics.py

Summary:

Module for diagnosing the Machine learning pipeline

Author: Arkaan Quanunga
Date: 16/02/2022

Functions:
- name:     model_predictions
- input:    data_path [str] The path to the dataset for predictions
- return:   predictions [list] The predictions of the dataset in a list format

- name:      dataframe_summary
- input:     data_path [str] The path to the dataset for analysis
- return:    summary_stats [dictionary] A dictionary of summary stats of each column of the dataset including:
                            - Mean
                            - Median
                            - Standard Deviation
                            - Inter Quartile Range

- name:     execution_time
- input:    file_names [list] A list of file names to run and calculate execution time
- return:   timings [dictionary] A dictionary with execution timings of all the files present in file_names list

- name: outdated_packages
- input: None
- return: indented results [] The list of outdated packages used by the machine learning models
"""
import sys

import pandas as pd
import numpy as np
import timeit
import pickle
import os
import json
from scipy import stats
import time
import subprocess as sp

# Load config.json and get environment variables

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
test_dataset_path = os.path.join(test_data_path, "testdata.csv")
ingested_dataset_path = os.path.join(dataset_csv_path, "finaldata.csv")


# Function to get model predictions
def model_predictions(data):
    """
    Read the deployed model and a test dataset, calculate predictions
    :param data_path: Path to the dataset for predictions [str]
    :return: predictions: List of predictions made by the deployed model [list]
    """
    # read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), "rb") as f:
        model = pickle.load(f)

    X_test = data.loc[
             :, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
             ]
    predictions = model.predict(X_test)

    return predictions  # value should be a list containing all predictions


# Function to get summary statistics
def dataframe_summary(data):
    """
    Calculates the summary of the dataset
    :param: data [dataframe] The dataset for the analysis
    :return: summary_stats [dictionary] list of statistics of the dataset:
                - mean
                - median
                - Standard deviation
                - InterQuartile Range (IQR)
    """

    # Reading the data

    X = data.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    # Calculating summary stats
    summary_stats = {col:
                         {"Mean": X[col].mean(axis=0), "Median": X[col].median(axis=0), "Standard_Deviation": X[col].std(axis=0),
                       "IQR": stats.iqr(X[col], interpolation='midpoint')} for col in X}

    return summary_stats  # return value should be a list containing all summary statistics
                          # Better it's a dictionary with extendable access

# Function for missing data
def dataframe_missing_data(data_file_path):
    """

    :param data_file_path: [str] The file_path of the data that is going to be checked for missing data
    :return: [float] Percentage of missing values
    """
    # Calculate percentage of missing values
    df = pd.read_csv(data_file_path)
    return df.isna().sum() / df.count().sum()

# Function to get timings
def execution_time(file_names):
    """
    Calculates the execution time for all the files present in the file_names list

    :param file_names [list] A list of file names to run and calculate execution time
    :return: timings [dictionary] A dictionary with execution timings of all the files present in file_names list
    """
    # calculate timing of training.py and ingestion.py
    timings ={}
    for file in file_names:
        start_time_training = time.time()
        sp.call(["python", file])
        end_time_training = time.time() - start_time_training
        timings[file]=end_time_training

    return timings
    # return a list of 2 timing values in seconds
    # Better use dictionary for easier access


# Function to check dependencies
def outdated_packages_list():
    # get a list of
    """
    Lists the outdated packages that are being used by the machine learning model
    :return: indented results [] The list of outdated packages
    """
    args = [sys.executable, "-m", "pip", "list", "--outdated"]
    results = sp.run(args, capture_output=True, check=True).stdout
    indented_results = ("\n" + results.decode().replace("\n", "\n   "))

    return indented_results


if __name__ == '__main__':
    # Running the script

    # Reading the dataset
    data = pd.read_csv(test_dataset_path)

    #  Model predictions
    model_predictions(data)

    #  Dataframe summary
    dataset = pd.read_csv(ingested_dataset_path)
    dataframe_summary(dataset)

    # Dataframe missing data
    dataframe_missing_data(ingested_dataset_path)

    # Defining the list of files to check execution time
    file_names = ["training.py", "ingestion.py", "scoring.py"]
    execution_time(file_names)

    # Checking for outdated packages
    outdated_packages_list()

