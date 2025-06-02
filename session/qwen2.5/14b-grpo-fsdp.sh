# CUDA_VISIBLE_DEVICES="7" trl vllm-serve --model mesolitica/Malaysian-Qwen2.5-14B-Reasoning-SFT --tensor_parallel_size 1

WANDB_PROJECT="fpf-Malaysian-Qwen2.5-14B-Reasoning-SFT-GRPO" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_HOME="/usr/local/cuda-12.8" \
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" \
torchrun --nproc_per_node 7 \
-m grpo_multitask \
--fsdp "full_shard auto_wrap" \
--fsdp_config fsdp.json \
--model_name_or_path mesolitica/Malaysian-Qwen2.5-14B-Reasoning-SFT \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--output_dir fpf-Malaysian-Qwen2.5-14B-Reasoning-SFT-GRPO \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--dataset huseinzol05/malaysian-reasoning-qa \
--logging_steps 1 \
--learning_rate 2e-6 \
--warmup_steps 10 \
--max_grad_norm 0.1 \
--save_steps 100 \
--save_total_limit 3 \
--rank 0 \
--ddp_find_unused_parameters false \
--use_liger_loss true \
--use_vllm true \
--max_prompt_length 256 \
--max_completion_length 2048 \
--num_generations 4 \
--vllm_server_port 8000 \
--top_k -1 \
--min_p 0.0 \
--temperature 1.0