import os
import requests
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"
with open("config.json", "r") as f:
    config = json.load(f)
output_model_path = os.path.join(config["output_model_path"])


#Call each API endpoint and store the responses
response1 = #put an API call here
response2 = #put an API call here
response3 = #put an API call here
response4 = #put an API call here

#combine all API responses
responses = #combine reponses here

#write the responses to your workspace



