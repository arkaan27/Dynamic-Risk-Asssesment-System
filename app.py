from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import create_prediction_model
import diagnosis 
import predict_exited_from_saved_model
from diagnostics import model_predictions, dataframe_summary,execution_time,outdated_packages_list, dataframe_missing_data
from reporting import score_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config["test_data_path"])
output_model_path = os.path.join(config["output_model_path"])
test_dataset_path = os.path.join(test_data_path, "testdata.csv")
model_path = os.path.join(output_model_path, "trainedmodel.pkl")
ingested_dataset_path = os.path.join(dataset_csv_path, "finaldata.csv")

with open(model_path, "rb") as f:
    prediction_model = pickle.load(f)


@app.route('/', methods = ['GET', 'OPTIONS'])
def hi():
    print("Welcome to my Dynamic Risk Assessment System")

#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    dataset = request.args.get("dataset")
    df = pd.read_csv(dataset)
    predictions = model_predictions(df)
    return predictions #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    #check the score of the deployed model
    return str(score_model(test_dataset_path)) #add return value (a single F1 score number)

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():
    #check means, medians, and modes for each column
    dataset = request.args.get("dataset")
    return str(dataframe_summary(dataset))#return a list of all calculated summary statistics

#Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    #check timing and percent NA values
    Pipeline = ["ingestion.py","training.py", "diagnostics.py", "deployment.py"]

    return {
        "Execution time in seconds" : execution_time(Pipeline),
        "Missing values in data": dataframe_missing_data(ingested_dataset_path),
        "outdated_packages_list": outdated_packages_list()
    }

    #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
