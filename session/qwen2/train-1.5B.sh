WANDB_PROJECT="finetune-Qwen2-1.5B" \
torchrun --nproc_per_node 4 \
-m train \
--model_name_or_path Qwen/Qwen2-1.5B \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 32 \
--output_dir finetune-qwen2 \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "/home/ubuntu/mosaic-qwen2-4096" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 4096 \
--save_steps 10 \
--save_total_limit 3 \
--gradient_checkpointing true \
--log_level "info" \
--torch_dtype "bfloat16"