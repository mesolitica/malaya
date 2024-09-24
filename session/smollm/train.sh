WANDB_PROJECT="finetune-HuggingFaceTB-SmolLM-360M" \
torchrun --nproc_per_node 4 \
-m train \
--model_name_or_path HuggingFaceTB/SmolLM-360M \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--output_dir finetune-SmolLM-360M \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 2 \
--train_file "combine-smollm" \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 4096 \
--save_steps 200 \
--save_total_limit 3 \
--gradient_checkpointing true \
--log_level "info" \
--torch_dtype "bfloat16"