import os
import requests
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"
with open("config.json", "r") as f:
    config = json.load(f)
output_model_path = os.path.join(config["output_model_path"])


#Call each API endpoint and store the responses
response1 = requests.post(
    f"{URL}prediction?dataset=testdata/testdata.csv").content #put an API call here
response2 = requests.get(f"{URL}scoring").content #put an API call here
response3 = requests.get(f"{URL}summary_stats").content #put an API call here
response4 = requests.get(f"{URL}diagnostics").content  #put an API call here

#combine all API responses
responses = [response1, response2, response3, response4] #combine reponses here

#write the responses to your workspace
file_to_save = os.path.join(output_model_path, "apireturns.txt")
with open(file_to_save, "w") as f:
    f.write(str(responses))


