"""
Name: scoring.py

Summary:
Takes a trained model, loads test data and calculates an F1 score for the model relative to test data

Saves the result in latestscore.txt file


Author: Arkaan Quanunga
Date: 16/02/2022

Functions:

1. score_model:

inputs:
- name:         test_data_path
  type:         [str]
  description:  The path to the test data

- name:         test_data_csv_name
- type:         [str]
- description:  The name of the dataset used for testing the model
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging
from datetime import datetime

# Initialising logger for checking steps
logging.basicConfig(
    filename='./logs/scoring.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Load config.json and get path variables

logging.info("Loading config.json for getting path variables")
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config["output_model_path"])


# Function for model scoring

def score_model(test_data_path, test_data_csv_name):
    """
    Takes a trained model, loads test data and calculates an F1 score for the model relative to test data
    :param test_data_path: [str] The path to the test data
    :param test_data_csv_name: [str] The name of the test dataset
    :return: Saves the result file as latestscore.txt file
    """

    # Defining necessary paths

    logging.info("Defining the necessary path variables")
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    test_dataset_path = os.path.join(test_data_path, test_data_csv_name)
    output_scores_path = os.path.join(output_model_path, "latestscore.txt")

    # Reading the dataset

    logging.info("Reading the dataset")
    df = pd.read_csv(test_dataset_path)

    # Loading the model

    logging.info("Loading the model with pickle module")
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Defining X variables and Y variables for testing

    logging.info("Defining X_test as lastmonth_activity, lastyear_activity and number_of_employees")
    X_test = df.loc[
             :, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
             ]

    logging.info("Defining y_test as exited from dataframe")
    y_test = df["exited"]

    # Getting the date and time for reference

    logging.info("Getting date for record keeping")
    dateTimeObj = datetime.now()
    date_now = str(dateTimeObj.day) + '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.year)

    # Making predictions

    logging.info("Making predictions using the Logistic Regression model")
    predictions = model.predict(X_test)

    logging.info("Generating the f1_score")
    f1_score = metrics.f1_score(predictions, y_test)

    # Creating all records as dictionary

    logging.info("Creating Records for predictions")
    allrecord = {"test_data_path": str(test_data_path),
                 "file_name": test_data_csv_name,
                 "Length_of_test_dataset":  str(len(df.index)),
                 "Date": date_now,
                 "F1_SCORE": str(f1_score)}

    # Writing the file

    logging.info("Writing file to latestscore.txt")

    with open(output_scores_path, "w") as f:
        for key,value in allrecord.items():
            f.write(key + '-> ' + value + "\n")


if __name__ == "__main__":
    score_model(test_data_path, "testdata.csv")
