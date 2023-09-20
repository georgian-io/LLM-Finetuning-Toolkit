FROM huggingface/transformers-pytorch-gpu:4.27.2 as base

WORKDIR /app

RUN pip install pip --upgrade
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt 

RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app 
ENV APP_MODEL_PATH="weights/checkpoints/assets"
ENV APP_MODEL_TYPE="causal"
ENV APP_TASK="summarization"
ENV APP_MAX_TARGET_LENGTH=50
ENV APP_TEMPERATURE="0.01"

COPY . /app
CMD uvicorn --host 0.0.0.0 --port 8080 --workers 2 api:app