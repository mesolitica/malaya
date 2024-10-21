WANDB_PROJECT="lora-embedding-256-llama3.2-1b-multipack" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 2 \
-m run-instruction-lora-embedding-multipack \
--model_name_or_path unsloth/Llama-3.2-1B-Instruct \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 6 \
--output_dir lora-embedding-256-llama3.2-1b-multipack \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file /home/husein/ssd4/continue-training/packing-4096 \
--logging_steps 1 \
--learning_rate 2e-5 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--block_size 24576 \
--save_steps 100 \
--save_total_limit 3 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16 \
--rank 256 \
--ddp_find_unused_parameters false \
--include_num_input_tokens_seen true