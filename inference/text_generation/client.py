import argparse
import requests
from text_generation import Client

def send_request(args):
    client = Client("http://0.0.0.0:8080")
    print(client.generate(args.prompt, max_new_tokens=args.max_new_tokens).generated_text)

    text = ""
    for response in client.generate_stream(args.prompt, max_new_tokens=args.max_new_tokens):
        if not response.token.special:
            text += response.token.text
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send a POST request to a FastAPI endpoint with a given prompt.")
    
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--max_new_tokens", type=int)
    
    args = parser.parse_args()
    
    generated_text = send_request(args.prompt)
    print(f"Prediction: {generated_text}")