from fastapi import FastAPI

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from predictor import Predictor

class Settings(BaseSettings):
    model_path: str = "weights/checkpoints/assets"
    model_type: str= "causal"
    task: str = "summarization"
    max_target_length: int = 50
    temperature: float = 0.01
    
    class Config:
        env_prefix = 'APP_'


class Payload(BaseModel):
    prompt: str


class Prediction(BaseModel):
    prediction: str


app = FastAPI()
settings = Settings()
predictor = Predictor(model_load_path=settings.model_path, model_type=settings.model_type,
                      task=settings.task)


@app.post("/predict", response_model=Prediction)
def predict(paylod: Payload) -> Prediction:
    prediction = predictor.predict(prompt=paylod.prompt, max_target_length=settings.max_target_length,
                                   temperature=settings.temperature)
    return Prediction(prediction=prediction)


@app.get("/health-check")
def health_check() -> str:
    return "ok"