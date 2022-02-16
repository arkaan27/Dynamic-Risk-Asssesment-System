
import pandas as pd
import numpy as np
import timeit
import pickle
import os
import json
import logging

# Initialising logger for checking steps
logging.basicConfig(
    filename='./logs/scoring.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# Load config.json and get environment variables

logging.info("Loading config.json for getting path variables")
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config["prod_deployment_path"])
test_dataset_path = os.path.join(test_data_path, "testdata.csv")

# Function to get model predictions
def model_predictions(data):
    """
    Read the deployed model and a test dataset, calculate predictions
    :param data: dataset for predictions [dataframe]
    :return: predictions: List of predictions made by the deployed model [list]
    """
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), "rb") as f:
        model = pickle.load(f)

    X_test = data.loc[
             :, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
             ]
    predictions = model.predict(X_test)

    return predictions #value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    return #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    return #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    return

if __name__ == '__main__':
    # Defining data
    data  = pd.read_csv(test_dataset_path)

    # Running the script
    model_predictions(data)
    # dataframe_summary()
    # execution_time()
    # outdated_packages_list()





    
