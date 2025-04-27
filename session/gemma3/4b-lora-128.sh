WANDB_PROJECT="lora-embedding-256-gemma3-4b-malaysian" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 8 \
-m train \
--model_name_or_path google/gemma-3-4b-it \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 3 \
--output_dir lora-embedding-256-gemma3-12b-malaysian \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file packing-4k \
--logging_steps 1 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--block_size 24576 \
--save_steps 100 \
--save_total_limit 3 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16 \
--rank 128 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4 \
--torch_compile