"""
Name: diagnosotics.py

Summary:

Module for diagnosing the Machine learning pipeline

Author: Arkaan Quanunga
Date: 16/02/2022

Functions:
- name:     model_predictions
- input:    data_path [str] The path to the dataset for predictions
- return:   predictions [list] The predictins of the dataset in a list format

- name:      dataframe_summary
- input:     data_path [str] The path to the dataset for analysis
- return:    summary_stats [dictionary] A dictionary of summary stats of each column of the dataset including:
                            - Mean
                            - Median
                            - Standard Deviation
                            - Inter Quartile Range

"""
import pandas as pd
import numpy as np
import timeit
import pickle
import os
import json
from scipy import stats

# Load config.json and get environment variables

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
test_dataset_path = os.path.join(test_data_path, "testdata.csv")
ingested_dataset_path = os.path.join(dataset_csv_path, "finaldata.csv")


# Function to get model predictions
def model_predictions(data_path):
    """
    Read the deployed model and a test dataset, calculate predictions
    :param data_path: Path to the dataset for predictions [str]
    :return: predictions: List of predictions made by the deployed model [list]
    """
    # read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), "rb") as f:
        model = pickle.load(f)

    # Reading the dataset
    data = pd.read_csv(data_path)
    X_test = data.loc[
             :, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
             ]
    predictions = model.predict(X_test)

    return predictions  # value should be a list containing all predictions


# Function to get summary statistics
def dataframe_summary(data_path):
    """
    Calculates the summary of the dataset
    :param: data [dataframe] The dataset for the analysis
    :return: summary stats [list] list of statistics of the dataset:
                - mean
                - median
                - Standard deviation
                - InterQuartile Range (IQR)
    """
    # calculate summary statistics here

    # Reading the data
    data = pd.read_csv(data_path)
    X = data.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    summary_stats = {col:
                         {"Mean": X[col].mean(axis=0), "Median": X[col].median(axis=0), "Standard_Deviation": X[col].std(axis=0),
                       "IQR": stats.iqr(X[col], interpolation='midpoint')} for col in X}

    return summary_stats  # return value should be a list containing all summary statistics


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    return  # return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list():
    # get a list of
    return


if __name__ == '__main__':
    # Running the script
    model_predictions(test_dataset_path)
    dataframe_summary(ingested_dataset_path)
    # execution_time()
    # outdated_packages_list()
