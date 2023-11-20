import json

def load_json(path):
    with open(path, 'r') as file:
        json_file = json.load(file)
    return json_file