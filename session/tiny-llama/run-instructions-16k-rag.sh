WANDB_PROJECT=fpf-tinyllama-1.1b-hf-instructions-16k-rag \
deepspeed run-instructions.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path mesolitica/tinyllama-1.1b-4096-fpf \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 6 \
--output_dir fpf-1.1b-instructions-16k-call \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 3 \
--train_file 'mosaic-chat-instructions-rag' \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 16384 \
--save_steps 200 \
--save_total_limit 2 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16