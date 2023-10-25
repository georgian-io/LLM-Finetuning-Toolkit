import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

summarizer_runner = bentoml.models.get("text-generation:latest").to_runner()

svc = bentoml.Service(
    name="text-generation", runners=[summarizer_runner]
)

class Prompt(BaseModel):
    text: str

input_spec = JSON(pydantic_model=Prompt)

@svc.api(input=input_spec, output=bentoml.io.Text())
async def generate(input_data: Prompt) -> str:
    generated = await summarizer_runner.async_run(input_data.dict()['text'], max_length=3000)
    return generated[0]["generated_text"]
