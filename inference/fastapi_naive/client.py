import argparse
import requests

def send_request(url, prompt):
    response = requests.post(url, json={"prompt": prompt})
    
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        response.raise_for_status()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a POST request to a FastAPI endpoint with a given prompt.")
    
    parser.add_argument("--url", type=str, default="http://0.0.0.0:8080/predict", help="Endpoint URL to make the POST request.")
    parser.add_argument("--prompt", type=str)
    
    args = parser.parse_args()
    
    prediction = send_request(args.url, args.prompt)
    print(f"Prediction: {prediction}")