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
import timeit

# Initialising logger for checking steps

logging.basicConfig(
    filename='./logs/training.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Load config.json and get path variables

logging.info("Loading config.json for getting path variables")
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


# Function for training the model
def train_model():
    # Use this logistic regression for training

    logging.info("Using Logistic regression for training")
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='ovr',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # Reading and separating variables

    logging.info("Reading the final.csv file")
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))

    logging.info("Defining X variables as lastmonth_activity, "
                 "lastyear_activity, "
                 "number_of_employees")
    X = df.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    logging.info("Defining Y variables as exited")
    y = df["exited"]

    # Splitting the data

    logging.info("Splitting the data by 80% for training and 20% for testing")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # fit the logistic regression to your data

    logging.info("Training the model")
    model = model.fit(X_train, y_train)

    # Creating model path directory

    logging.info("Creating the model path directory")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # write the trained model to your workspace in a file called trainedmodel.pkl

    logging.info("Saving the model in model_path")
    save_model_filename = os.path.join(model_path, "trainedmodel.pkl")
    pickle.dump(model, open(save_model_filename, "wb"))


if __name__ == "__main__":
    train_model()
