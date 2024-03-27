WANDB_PROJECT="finetune-gemma-2b" \
deepspeed train.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path google/gemma-2b \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--output_dir finetune-gemma-2b \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "/home/ubuntu/mosaic-gemma-8192" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 4096 \
--save_steps 50 \
--save_total_limit 3 \
--gradient_checkpointing true \
--log_level "debug" \
--torch_dtype "bfloat16"