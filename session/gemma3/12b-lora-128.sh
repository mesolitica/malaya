WANDB_PROJECT="lora-embedding-128-gemma3-12b-malaysian-8k" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_HOME="/usr/local/cuda-12.8" \
CHUNK_SIZE="32" \
torchrun --nproc_per_node 8 \
-m train \
--deepspeed ds_config_zero3.json \
--model_name_or_path google/gemma-3-12b-it \
--per_device_train_batch_size 10 \
--gradient_accumulation_steps 1 \
--output_dir lora-embedding-128-gemma3-12b-malaysian-8k \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file packing-8k \
--logging_steps 1 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--block_size 24576 \
--save_steps 50 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16 \
--rank 128 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4