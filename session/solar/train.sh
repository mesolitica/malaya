WANDB_PROJECT="finetune-solar" \
deepspeed train.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path upstage/SOLAR-10.7B-v1.0 \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 1 \
--output_dir finetune-solar \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "/home/ubuntu/mosaic-solar" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 4096 \
--save_steps 200 \
--save_total_limit 3 \
--gradient_checkpointing true \
--log_level "info" \
--torch_dtype "bfloat16"