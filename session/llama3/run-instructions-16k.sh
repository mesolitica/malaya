WANDB_PROJECT=fpf-llama-3-8b-8192-hf-packing \
deepspeed run-instruction-packing.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path mesolitica/llama-3-8b-8192-hf \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 3 \
--output_dir fpf-llama-3-8b-8192-hf \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 5 \
--train_file 'mosaic-chat-instruction-v6-llama3-16k' \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 16384 \
--save_steps 50 \
--save_total_limit 2 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16