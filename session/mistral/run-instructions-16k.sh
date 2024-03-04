WANDB_PROJECT=fpf-mistral-7b-hf-instructions-16k-v5 \
deepspeed run-instruction-packing.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path mesolitica/malaysian-mistral-7b-32k-instructions-v5 \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 1 \
--output_dir fpf-7b-instructions-16k-call \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 5 \
--train_file 'mosaic-chat-instruction-v5-mistral-16k' \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 16384 \
--save_steps 50 \
--save_total_limit 2 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16