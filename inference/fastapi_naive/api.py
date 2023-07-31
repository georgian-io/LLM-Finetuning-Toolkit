from fastapi import FastAPI

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from fastapi_naive.predictor import Predictor

# Settings for summarization
class Settings(BaseSettings):
    model_path: str = "weights/checkpoints-for-summarization/assets"
    task: str = "summarization"
    max_target_length: int = 50


# Settings for classification
# class Settings(BaseSettings):
#     model_path: str = 'weights/checkpoints-for-classification/assets'
#     task: str = 'classification'
#     max_target_length: int = 20


class Payload(BaseModel):
    prompt: str


class Prediction(BaseModel):
    prediction: str


app = FastAPI()
settings = Settings()
predictor = Predictor(model_load_path=settings.model_path, task=settings.task)


@app.post("/predict", response_model=Prediction)
def predict(paylod: Payload) -> Prediction:
    prediction = predictor.predict(prompt=paylod.prompt, max_target_length=settings.max_target_length)
    return Prediction(prediction=prediction)


@app.get("/health-check")
def health_check() -> str:
    return "ok"