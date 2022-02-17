
import os
import json
import subprocess as sp
import training
import scoring
import deployment
import diagnostics
import reporting

from ast import literal_eval

from scoring import score_model

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = os.path.join(config["input_folder_path"])
dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
ingested_file_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
latest_score_path = os.path.join(prod_deployment_path, "latestscore.txt")

# Check and read new data
# first, read ingestedfiles.txt
with open(ingested_file_path, "r") as f:
    ingested_files = literal_eval(f.read())
# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files_found = False
for f in os.listdir(input_folder_path):
    if f not in ingested_files:
        print(f)
        new_files_found = True

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if new_files_found:
    sp.call(["python", "ingestion.py"])
else:
    print("No new data ingested, exiting....")
    exit()

# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(latest_score_path, "r") as f:
    latest_score = float(f.read())

score_model = score_model(dataset_csv_path, "finaldata.csv")

print(latest_score, score_model)

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
print("Proceed: Model drift found") if score_model < latest_score else exit()

sp.call(["python", "training.py"])
# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
sp.call(["python", "deployment.py"])
# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
sp.call(["python", "diagnostics.py"])
sp.call(["python", "apicalls.py"])
sp.call(["python", "reporting.py"])



