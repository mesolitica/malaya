WANDB_PROJECT=Qwen-Qwen2.5-7B-Instruct-qlora-32k \
deepspeed run-instruction-packing-qlora.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path Qwen/Qwen2.5-7B-Instruct \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 6 \
--output_dir fpf-0.5-instructions-16k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 5 \
--train_file 'malaysian-qwen2.5-32k' \
--logging_steps 1 \
--learning_rate 2e-5 \
--block_size 32768 \
--save_steps 50 \
--save_total_limit 3 \
--gradient_checkpointing true \
--neftune_noise_alpha 5.0 \
--torch_dtype bfloat16