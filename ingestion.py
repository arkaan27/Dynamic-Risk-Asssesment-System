"""
Name: Ingestion.py

Summary:

Automation Script for extracting unstructured data from one or multiple sources.
Then preeparing the data for training machine learning model.
     
Author: Arkaan Quanunga
Date: 16/02/2022
    
Functions:

1. merge_multiple_dataframe: check for datasets, compile them together, and write to an output file
2. main: running the script

"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import logging

# Initialising logger for checking steps
logging.basicConfig(
    filename='./logs/ingestion.logs',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    """
    check for datasets, compile them together, and write to an output file
    """

    current_dir = os.getcwd()
    input_path = f"{current_dir}/{input_folder_path}"

    # Creating the output folder if it does not exist
    logging.info("Creating output directory")
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Joining the output folder path
    output_path = os.path.join(os.getcwd(), output_folder_path)

    # Listing the files present in the input path
    files = os.listdir(input_path)

    # Checking only .csv files
    logging.info("Checking for only .csv files")
    datasets = [x for x in files if x[-4:] == '.csv']

    # Creating empty dataframe

    logging.info("Creating an empty dataframe")
    combined_df = pd.DataFrame()


    # Reading the data frame from the folder

    logging.info("Reading the files with pandas module")
    for dataset in datasets:
        df = pd.read_csv(os.path.join(input_path, dataset))
        logging.info("Appending the newly created dataframe combined_df")
        combined_df= combined_df.append(df)

    # Getting the date and time for reference

    logging.info("Getting date for record keeping")
    dateTimeObj = datetime.now()
    date_now = str(dateTimeObj.day) + '/' + str(dateTimeObj.month) + '/' + str(dateTimeObj.year)

    # # Combining the files into one dataframe
    #
    # logging.info("Combining the dataframes together")
    # combined_df = pd.concat(df)

    # Cleaning the data

    logging.info("Dropping duplicates")
    combined_df.drop_duplicates(inplace=True)

    # Creating all records

    logging.info("Creating Records for dataframe")
    allrecord = {
        "input_folder_path": input_folder_path,
        "List_of_datasets": str(files),
        "Length_of_combined_dataset": str(len(combined_df.index)),
        "Date": date_now}
    # Exporting the dataframe to csv

    logging.info("Exporting the dataframe to csv as finaldata.csv")
    output_csv_path = os.path.join(output_path, "finaldata.csv")
    combined_df.to_csv(output_csv_path, index=False)

    # Writing ingested file.txt for file path

    logging.info("Adding records to ingested_files.txt")
    output_txt_path = os.path.join(output_path, "ingestedfiles.txt")
    with open(output_txt_path, "w") as f:
        for key, value in allrecord.items():
            f.write(
                key + "-> " + value + "\n",
            )


if __name__ == '__main__':
    merge_multiple_dataframe()
