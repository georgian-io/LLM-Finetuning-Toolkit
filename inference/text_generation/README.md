# Text Generation Inference

## Create a standalone model

In order to use HuggingFace Text Generation Inference, we need to merge trained model with LoRa layers into the standalone one.

1. Log in into your HuggingFace account with created model repository. 
   ```
   huggingface-cli login
   ```
2. Run the merge script.
   ```
   python3 merge_script.py --repo_id your_model_repository
   ```
3. Define next variables in the CLI.
   ```
   model=your_model_repository
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
   token=<your huggingface token>
   ```
4. Run Docker container.
   ```
   docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.0 --model-id $model
   ```
## Use for inference

1. Make requests through CLI.
   
   ```
   curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"summarize: Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last time we were at the park together\nHannah: I don't know him well\nHannah: <file_gif>\nAmanda: Don't be shy, he's very nice\nHannah: If you say so..\nHannah: I'd rather you texted him\nAmanda: Just text him 🙂\nHannah: Urgh.. Alright\nHannah: Bye\nAmanda: Bye bye","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'
   ```