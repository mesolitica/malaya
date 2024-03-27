WANDB_PROJECT="finetune-Qwen1.5-0.5B" \
~/.local/bin/deepspeed train.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path Qwen/Qwen1.5-0.5B \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--output_dir finetune-Qwen1.5-0.5B \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "/home/ubuntu/mosaic-qwen2" \
--logging_steps 1 \
--learning_rate 2e-5 \
--save_steps 200 \
--save_total_limit 3 \
--gradient_checkpointing true \
--log_level "info" \
--torch_dtype "bfloat16"