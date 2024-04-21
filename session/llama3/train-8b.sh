WANDB_PROJECT="finetune-llama-3-8b" \
deepspeed train.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path meta-llama/Meta-Llama-3-8B \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--output_dir finetune-llama-3-8b \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "/home/ubuntu/mosaic-llama-3-8192" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 8192 \
--save_steps 50 \
--save_total_limit 3 \
--gradient_checkpointing true \
--log_level "debug" \
--torch_dtype "bfloat16"