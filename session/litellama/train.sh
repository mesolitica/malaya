WANDB_PROJECT="finetune-LiteLlama-460M-1T" \
torchrun --nproc_per_node 8 \
-m train \
--model_name_or_path ahxt/LiteLlama-460M-1T \
--per_device_train_batch_size 24 \
--gradient_accumulation_steps 1 \
--output_dir finetune-LiteLlama-460M-1T \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "/home/ubuntu/mosaic-litellama" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 4096 \
--save_steps 200 \
--save_total_limit 3 \
--gradient_checkpointing true \
--log_level "info" \
--torch_dtype "bfloat16"