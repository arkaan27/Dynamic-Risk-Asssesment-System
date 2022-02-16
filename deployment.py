"""
Name: deployment.py

Summary:
Module for deploying the model

Author: Arkaan Quanunga
Date: 16/02/2022

Function:
store_model_into_pickle
:return: Saves all the files of production model, including model.pkl,ingestedfiles and F1_score
"""
import shutil

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

# Initialising logger for checking steps

logging.basicConfig(
    filename='./logs/deployment.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Load config.json and correct path variable

logging.info("Loading config.json for getting path variables")
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path'])

# Model
model_to_copy = os.path.join(output_model_path, "trainedmodel.pkl")
scores_to_copy = os.path.join(output_model_path, "latestscore.txt")
ingestedfiles_to_copy = os.path.join(dataset_csv_path, "ingestedfiles.txt")
copy_all_files = [model_to_copy, scores_to_copy, ingestedfiles_to_copy]


# Function for deployment

def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    # Creating the production deployment model path

    if not os.path.exists(prod_deployment_path):
        logging.info("Creating production deployment directory")
        os.makedirs(prod_deployment_path)

    # Copying all files to production deployment path

    logging.info("Copying all files to production deployment path")
    for f in copy_all_files:
        shutil.copy(f, prod_deployment_path)


if __name__ == "__main__":
    store_model_into_pickle()
