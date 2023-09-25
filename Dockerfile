FROM python:3.9-slim

WORKDIR /code

COPY . .

RUN pip install --upgrade -r requirements.txt --no-cache-dir

ENTRYPOINT [ "./finetune.sh" ]