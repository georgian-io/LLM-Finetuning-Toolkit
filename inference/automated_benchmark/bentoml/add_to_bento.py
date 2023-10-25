import bentoml
import transformers

task = "text-generation"
model = "mariiaponom/redp_3b_class"
bentoml.transformers.save_model(
    task,
    transformers.pipeline(task, model=model),
    metadata=dict(model_name=model),
)