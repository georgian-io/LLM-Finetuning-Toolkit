# Deployment

In this section you can find the instructions on how to deploy your model using FastApi and Text Generation Inference. 

To follow these instructions you need:

- Docker installed
- Path of the folder with model weights
- HuggingFace account

Note: To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). 

## Automated deployment and benchmark

With automated deployment you can easily deploy LLama-2, RedPajama, Falcon or Flan model and load test it for different number of requests. You just need a published repo with a model on HuggingFace (for vLLM and TGI) or a folder with model files (for Ray).

1. Go to <code> automated_benchmark </code> folder 
2. Run command to start the server:

   ```
   python run_inference.py
   ```
3. Once the server is started, run command for benchmark in a separate window:

   ```
   python run_benchmark.py
   ```
4. After executing you can build a plot to compare performance of difference servers for specific model:

   ```
   cd benchmark_results/plots/
   python create_plot --model_name llama_7b_class --compute a10 --title "Example title"
   ```


## Manual deployment 

### FastApi

For building FastApi application, do the following:


1. Copy folder with model weights to the ```fastapi_naive``` directory
   
   ```
   cp model_weights llm-tuning-hub/inference/fastapi_naive 
   ```
2. Navigate to Inference folder
   
   ```
   cd ./inference 
   ```
3. Build the Docker image
    ```
    docker build -t fastapi_ml_app:latest ./fastapi_naive/
    ```
4. Run Docker image specifying the parameters:

   - <code> APP_MODEL_PATH </code>: model path of your model (the one from the step 1)
   - <code> APP_TASK </code>: summarization/classification depending for what task your model was trained on
   - <code> APP_MAX_TARGET_LENGTH </code>: the maximum numbers of tokens to generate, ignoring the number of tokens in the prompt
   - <code> APP_MODEL_TYPE </code>: depending on what model you want to deploy, you should choose respective model type according to this table
  
        | Model      | Type    |
        |------------|---------|
        | Flan-T5       | seq2seq |
        | Falcon-7B     | causal  |
        | RedPajama  | causal  |
        | LLama-2      | causal  |
    <p></p>

   ```
   docker run --gpus all -it --rm -p 8080:8080 --name app-web-test-run-ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all -e APP_MODEL_PATH="weights/checkpoints-for-summarization/assets" -e APP_MODEL_TYPE="causal" -e APP_TASK="summarization" -e APP_MAX_TARGET_LENGTH=100 fastapi_ml_app:latest
   ```
5. Test application
   
   ```
   python client.py --url http://localhost:8080/predict --prompt "Your custom prompt here"
   ```

### [Text Generation Inference](https://github.com/huggingface/text-generation-inference)

1. Install HuggingFace library:

    ```
    pip install huggingface_hub
    ```

2. Login into your HuggingFace account:
   
   ```
   huggingface-cli login
   ```

    Note: you will need a read/write token which you can create in Settings in your HF account. 

3. Create [New model](https://huggingface.co/new) repository in HugginFace
4. For using Text Generation Inference you need standalone model which you can get using merge script:
   
   ```
   python merge_script.py --model_path /my/path --model_type causal --repo_id johndoe/new_model
   ```
5. Serve the model:
   
   ```
   model=meta-llama/Llama-2-7b-chat-hf
   volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
   token=<your cli READ token>
   ```

   ```
   docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id $model
   ```
   
### [vLLm](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)

1. Install the package:
   
   ```
   pip install vllm
   ```
2. Start the server:
   
   Use the model name from HuggingFace repository for ```--model``` argument

   ```
   python -m vllm.entrypoints.openai.api_server --model username/model
   ```
3. Make request:

   ```
   curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
      "model": "facebook/opt-125m",
      "prompt": "San Francisco is a",
      "max_tokens": 7,
      "temperature": 0
      }'
   ```