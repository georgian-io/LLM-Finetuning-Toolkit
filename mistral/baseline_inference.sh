#python mistral_baseline_inference.py --task_type classification --prompt_type zero-shot & wait
#python mistral_baseline_inference.py --task_type classification --prompt_type few-shot & wait
python mistral_baseline_inference.py --task_type summarization --prompt_type zero-shot --use_flash_attention & wait
python mistral_baseline_inference.py --task_type summarization --prompt_type few-shot --use_flash_attention
