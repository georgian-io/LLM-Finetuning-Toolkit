import requests
import json

# Specify the URL
url = "http://0.0.0.0:8000/"

# Define the payload
text = '{"text":"Classify the following sentence that is delimited with triple backticks. ### Sentence: I just found out about the sublinguals disappearing too I dont know why Perhaps because they werent as profitable as cafergot Too bad since tablets are sometimes vomited up by migraine patients and they dont do any good flushed down the toilet I suspect well be moving those patients more and more to the DHE nasal spray which is far more effective Gordon Banks N3JXP Skepticism is the chastity of the intellect and gebcadredslpittedu it is shameful to surrender it too soon ### Class:"}'


# Send the POST request
response = requests.post(url, json=text)

# Check the response
if response.status_code == 200:
    print("Request was successful:", response.text)
else:
    print("Failed to retrieve data:", response.status_code, response.text)