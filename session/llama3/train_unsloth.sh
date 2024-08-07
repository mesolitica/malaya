WANDB_PROJECT="unsloth-malaysian-llama-3-8b-instruct-16k" \
~/.local/bin/torchrun --nproc_per_node 4 \
-m train_unsloth \
--model_name_or_path mesolitica/malaysian-llama-3-8b-instruct-16k \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 3 \
--output_dir unsloth-malaysian-llama-3-8b-instruct-16k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 5 \
--dataset 'final-sft-llama3-packing-32k' \
--logging_steps 1 \
--learning_rate 5e-5 \
--embedding_learning_rate 5e-6 \
--context_length 32768 \
--save_steps 10 \
--save_total_limit 2 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--optim 'adamw_8bit' \
--warmup_ratio 0.1 \
--lr_scheduler_type 'cosine'