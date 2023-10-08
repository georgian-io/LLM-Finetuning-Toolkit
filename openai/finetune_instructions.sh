# Upload data for summarization task
python gpt_finetune.py --job_type upload_data --task_type summarization 

# Upload data for classification task
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.025
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.05
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.1
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.25
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.5
python gpt_finetune.py --job_type upload_data --task_type classification --train_sample_fraction 0.99

# Submit finetuning job for summarization
python gpt_finetune.py --job_type submit_job --epochs 1 --model gpt-3.5-turbo --training_file_id file-6IuuVdURWMQu4Xox0GVKDvML

# Submit finetuning job for classification

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
