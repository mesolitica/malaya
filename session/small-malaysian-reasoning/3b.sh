WANDB_PROJECT="lora-embedding-256-llama3.2-3b-small-malaysian-reasoning" \
CUDA_VISIBLE_DEVICES="0,1" \
TORCH_DISTRIBUTED_DEBUG="info" \
torchrun --nproc_per_node 2 \
-m train \
--model_name_or_path unsloth/Llama-3.2-3B-Instruct \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 6 \
--output_dir lora-embedding-256-llama3.2-3b-small-malaysian-reasoning \
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
--rank 256 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 3 \
--dataloader_prefetch_factor 4