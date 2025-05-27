# CUDA_VISIBLE_DEVICES="7" trl vllm-serve --model mesolitica/Malaysian-Qwen2.5-7B-Reasoning-SFT --tensor_parallel_size 1

WANDB_PROJECT="fpf-Malaysian-Qwen2.5-7B-Reasoning-SFT-GRPO" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_HOME="/usr/local/cuda-12.8" \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" \
torchrun --nproc_per_node 7 \
-m grpo \
--deepspeed ds_config_zero3.json \
--model_name_or_path mesolitica/Malaysian-Qwen2.5-7B-Reasoning-SFT \
--per_device_train_batch_size 3 \
--gradient_accumulation_steps 2 \
--output_dir fpf-Malaysian-Qwen2.5-7B-Reasoning-SFT-GRPO \
--bf16 --do_train --do_eval false --num_train_epochs 2 \
--dataset huseinzol05/malaysian-dialect-qa \
--logging_steps 1 \
--learning_rate 2e-6 \
--warmup_steps 10 \
--max_grad_norm 0.01 \
--weight_decay 0.01 \
--save_steps 50 \
--gradient_checkpointing true \
--rank 0 \
--ddp_find_unused_parameters false \
--use_liger_loss false \
--use_vllm true \
--max_prompt_length 256 \
--max_completion_length 8192 \
--num_generations 42 \
--vllm_server_port 8000 \
--top_k -1 \
--min_p 0.0 \
--temperature 1.0