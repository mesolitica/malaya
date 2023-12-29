WANDB_PROJECT="finetune-tinyllama-1.1B" \
torchrun --nproc_per_node 4 \
-m train \
--model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--per_device_train_batch_size 24 \
--gradient_accumulation_steps 1 \
--output_dir finetune-tinyllama \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "/home/ubuntu/mosaic-tinyllama" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 4096 \
--save_steps 200 \
--save_total_limit 3 \
--gradient_checkpointing true \
--deepspeed ds_config_zero3.json \
--log_level "info"
