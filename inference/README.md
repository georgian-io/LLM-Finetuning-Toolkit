# Deployment

In this section you can find the instructions on how to deploy your models using different inference servers.

## Prerequisites

### General 

To follow these instructions you need:

- Docker installed
- HuggingFace repository with a merged model (follow steps 1-4 from [How to merge the model](#how-to-merge-the-model))

Note: To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). 

### Load testing

- [Vegeta](https://github.com/tsenart/vegeta) installed (follow [this guide](https://geshan.com.np/blog/2020/09/vegeta-load-testing-primer-with-examples/) for installation)



## Automated deployment and benchmark

With automated deployment you can easily deploy LLama-2, RedPajama, Falcon or Flan models and load test them for different number of requests. 

Go to <code> automated_deployment </code>folder.

```
cd automated_deployment
```

### Deployment

Before running the inference, you will need to fill the <code>config.json</code> file which has the next default structure:

```
{
    "server": "tgi",  
    "huggingface_repo": "NousResearch/Llama-2-7b-hf",
    "huggingface_token": "",
    "model_type": "llama",
    "max_tokens": 20
}
```

#### server

   Mappings for the possible servers you can deploy on:

   | Server | Parameter name |
   |-----------------|-----------------|
   | vLLM     | ```vllm```     |
   | Text Generation Inference     | ```tgi```     |
   | Ray     | ```ray```     |
   |Triton Inference Server with vLLM backend | ```triton_vllm```|


#### huggingface_token

Read/Write token for your HuggingFace account.

#### huggingface_repo

The model repository on HuggingFace that stores model files. Pass in the format ```username/repo_name```. 

#### max_tokens

Maximum number of tokens you model should generate (should be integer value). 

#### model_type

Mappings for different model types.
   | Model      | Type    |
   |------------|---------|
   | Flan-T5       | flan |
   | Falcon-7B     | falcon  |
   | RedPajama  | red_pajama  |
   | LLama-2      | llama  |


After modifying the fields according to your preferences, run next command to start the server:

```
python run_inference.py
```



### Send request to the server

When the server has starter, you now are able to send the request. 

1. Run the following command:

```
python send_post_request.py inference
```
2. You will be asked then to provide the input.

For example:

```
Input: Classify the following sentence that is delimited with triple backticks. ### Sentence:I was wondering if anyone out there could enlighten me on this car I saw the other day. It was a 2-door sports car, looked to be from the late 60s/ early 70s. It was called a Bricklin. The doors were really small. In addition, the front bumper was separate from the rest of the body. This is all I know. If anyone can tellme a model name, engine specs, years of production, where this car is made, history, or whatever info you have on this funky looking car, please e-mail. ### Class:
```

### Benchmark

If you want to find out what latency, thoughput each server provides you can perform the benchmark using [Vegeta](https://github.com/tsenart/vegeta) load-testing tool.

We currently support benchmark for classification/summarization tasks.

Before running the command you will have to add few more fields to the `config.json`:
```
{
    ...

    "task": "classification",
    "model_name": "llama_7b_class",
    "duration": "10s",
    "rate": "10"
}  
```
#### task

You should specify task your model was trained for, either ```classification``` or ```summarization```.

#### model_name

Text identifier of the model for summary table (can be anything).

#### duration and rate

Duration of the benchmark test. During each second certain name of requests (rate value) will be sent. If the duration is `10s` and rate is `20`, in total `200` requests will be sent.

Usually with longer time you will be able to send less requests per second without the server crashing.

Once the server is started, run command for benchmark in a separate window:

   ```
   python run_benchmark.py
   ```

The test will run 2 times for more fair results and in the end all metrics will be calculated with deviation.

<b> Raw data (Vegeta output for 1 test) </b>

```
Requests      [total, rate, throughput]         100, 10.10, 9.87
Duration      [total, attack, wait]             10.137s, 9.9s, 236.754ms
Latencies     [min, mean, 50, 90, 95, 99, max]  227.567ms, 347.64ms, 325.601ms, 421.165ms, 424.789ms, 426.472ms, 426.884ms
Bytes In      [total, mean]                     3200, 32.00
Bytes Out     [total, mean]                     36900, 369.00
Success       [ratio]                           100.00%
Status Codes  [code:count]                      200:100
Error Set:
```

<b> Processed data (summary of results for 2 tests)</b>
| model          | server | rps | latency_with_deviation | throughput_with_deviation | duration_with_deviation |
|----------------|--------|-----|-----------------------|---------------------------|-------------------------|
| llama_7b_class | tgi    | 10.1| 0.465±0.315           | 7.200±3.600               | 10.207±0.228            |




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

### [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)

#### How to merge the model

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
#### Serve the model with TGI:
   
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