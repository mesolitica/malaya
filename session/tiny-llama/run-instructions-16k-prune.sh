WANDB_PROJECT=fpf-prune-tinyllama-1.1b-hf-instructions-16k \
deepspeed run-instruction-packing.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path huseinzol05/570M-tinyllama \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--output_dir fpf-prune-1.1b-instructions-16k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file 'mosaic-chat-instruction-v5-tinyllama-16k' \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 16384 \
--save_steps 200 \
--save_total_limit 2 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16