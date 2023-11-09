from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel
import json
import typer

def main(huggingface_repo: str, huggingface_token: str, aws_role: str):
    llm_image = get_huggingface_llm_image_uri(
        "huggingface",
        version="1.0.3"
    )

    instance_type = "ml.g5.4xlarge"
    number_of_gpu = 1
    health_check_timeout = 300
   
    config = {
        'HF_MODEL_ID': huggingface_repo,
        'HF_API_TOKEN': huggingface_token,
        'SM_NUM_GPUS': json.dumps(number_of_gpu), 
        'MAX_INPUT_LENGTH': json.dumps(1024),  
        'MAX_TOTAL_TOKENS': json.dumps(2048), 
    }

    llm_model = HuggingFaceModel(
        role=aws_role,
        image_uri=llm_image,
        env=config
    )

    llm_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        container_startup_health_check_timeout=health_check_timeout,
    )

if __name__ == "__main__":
    typer.run(main)