WANDB_PROJECT="lora-embedding-64-audio-qwen2.5-7b-malaysian-10k-stage2" \
TORCH_DISTRIBUTED_DEBUG="info" \
CUDA_HOME="/usr/local/cuda-12.8" \
CHUNK_SIZE="32" \
torchrun --nproc_per_node 8 \
-m qwen_audio_stage2 \
--deepspeed ds_config_zero3.json \
--model_name_or_path mesolitica/Malaysian-Audio-Qwen2.5-7B-Instruct \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 1 \
--output_dir lora-embedding-64-audio-qwen2.5-7b-malaysian-10k-stage2 \
--bf16 --do_train --do_eval false --num_train_epochs 5 \
--train_file speech-packing-10k \
--logging_steps 1 \
--learning_rate 2e-5 \
--lr_scheduler_type linear \
--warmup_steps 50 \
--weight_decay 0.01 \
--block_size 10240 \
--save_steps 100 \
--save_total_limit 5 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16 \
--rank 64 \
--ddp_find_unused_parameters false \
--dataloader_num_workers 5 \
--dataloader_prefetch_factor 4