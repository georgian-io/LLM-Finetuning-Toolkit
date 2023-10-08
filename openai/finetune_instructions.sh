# Phase 1: Upload data to OpenAI

# Summarization
python gpt_finetune.py --job_type upload_data --task_type summarization 

# Classification
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.025
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.05
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.1
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.25
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.5
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.99


# Phase 2: Submit finetuning job to OpenAI

# Summarization
python gpt_finetune.py --job_type submit_job --epochs 1 --model gpt-3.5-turbo --training_file_id file-6IuuVdURWMQu4Xox0GVKDvML

# Classification

# 0.025
python gpt_finetune.py --job_type submit_job --epochs 5 --model gpt-3.5-turbo --training_file_id file-fUSQKZSOh2Y9J1IVTIUMHrEf 

# 0.05
python gpt_finetune.py --job_type submit_job --epochs 5 --model gpt-3.5-turbo --training_file_id file-rd0OjxW0pvfTQFLqBgszsoG2

# 0.1
python gpt_finetune.py --job_type submit_job --epochs 5 --model gpt-3.5-turbo --training_file_id file-wliGSsJ71POdyR71RfNgV4lQ

# 0.25
python gpt_finetune.py --job_type submit_job --epochs 5 --model gpt-3.5-turbo --training_file_id file-vwI4frnPUBCbayU45A2VrFff

# 0.5
python gpt_finetune.py --job_type submit_job --epochs 5 --model gpt-3.5-turbo --training_file_id file-rRYatOIdsOeo00t3yotarpsP

# 0.99
python gpt_finetune.py --job_type submit_job --epochs 5 --model gpt-3.5-turbo --training_file_id file-q8E4UMMC7bEr1lRf0EVbBVDA



# Phase 3: Infer finetuned model

# Summarization
python gpt_finetune.py --job_type infer_finetuned_model --task_type summarization --model_id ft:gpt-3.5-turbo-0613:georgian::87Dj76DB

# Classification

# 0.025
python gpt_finetune.py --job_type infer_finetuned_model --task_type classification --model_id ft:gpt-3.5-turbo-0613:georgian::87Dq5Smr

# 0.05
python gpt_finetune.py --job_type infer_finetuned_model --task_type classification --model_id ft:gpt-3.5-turbo-0613:georgian::87EUhhjZ

# 0.1
python gpt_finetune.py --job_type infer_finetuned_model --task_type classification --model_id ft:gpt-3.5-turbo-0613:georgian::87EjZ6Me

# 0.25
python gpt_finetune.py --job_type infer_finetuned_model --task_type classification --model_id ft:gpt-3.5-turbo-0613:georgian::87EoCdcG

# 0.5
python gpt_finetune.py --job_type infer_finetuned_model --task_type classification --model_id ft:gpt-3.5-turbo-0613:georgian::87FzhTQt

# 0.99
python gpt_finetune.py --job_type infer_finetuned_model --task_type classification --model_id ft:gpt-3.5-turbo-0613:georgian::87GsKvdl

