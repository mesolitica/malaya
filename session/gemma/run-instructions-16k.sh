WANDB_PROJECT=gemma-2B-8192-fpf-instructions-16k \
deepspeed run-instruction-packing.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path mesolitica/gemma-2B-8192-fpf \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--output_dir gemma-2B-8192-fpf-instructions-16k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 5 \
--train_file 'mosaic-chat-instruction-v5-gemma-16k' \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 16384 \
--save_steps 50 \
--save_total_limit 3 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16