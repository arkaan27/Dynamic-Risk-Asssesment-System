"""
_summary_

Ingestion     
    
Author: Arkaan Quanunga
Date: 16/02/2022
    
Functions:

Automation Script for extracting unstructured data from one or multiple sources.
Then preeparing the data for training machine learning model.

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

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file

    current_dir = os.getcwd()
    input_path= f"{current_dir}/{input_folder_path}"
    
    # Creating the output folder if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        logging.info("Creating output directory")
        
    # Joining the output folder path
    output_path = os.path.join(os.getcwd, output_folder_path)
    
    # Listing the files present in the input path
    files = os.listdir(input_path)
    
    # Reading the data frame from the folder
    
    logging.info("Reading the files with pandas module")
    df = [pd.read_csv(os.path.join(input_path,fname) for fname in files)]
    
    # Combining the files into one dataframe
    
    logging.info("Combining the dataframes together")
    combined_df = pd.concat(df)
    
    # Cleaning the data
    
    logging.info("Dropping duplicates")
    combined_df.drop_duplicates(inplace=True)
    
    
    # Exporting the dataframe to csv
    
    logging.info("Exporting the dataframe to csv as finaldata.csv")
    output_csv_path = os.path.join(output_path,"finaldata.csv")
    combined_df.to_csv(output_csv_path,index=False)

    # Writing ingested file.txt for file path
    
    logging.info("Adding the files to ingestedfiles text")
    output_txt_path = os.path.join(output_path,"ingestedfiles.txt")
    with open(output_txt_path, "w") as f:
        f.write(str(files))

if __name__ == '__main__':
    merge_multiple_dataframe()
