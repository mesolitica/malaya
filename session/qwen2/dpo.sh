WANDB_PROJECT="dpo-malaysian-Qwen1.5-0.5B-16k-instructions" \
torchrun --nproc_per_node 8 \
-m dpo \
--model_name_or_path=mesolitica/malaysian-Qwen1.5-0.5B-16k-instructions \
--per_device_train_batch_size 8 \
--learning_rate 5e-7 \
--warmup_ratio 0.1 \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 10 \
--logging_steps 1 \
--save_steps 50 \
--save_total_limit 3 \
--gradient_checkpointing true \
--output_dir="dpo-qwen" \
--max_length 8192 \
--max_prompt_length 1024 \
--log_level "info" \
--torch_dtype "bfloat16"