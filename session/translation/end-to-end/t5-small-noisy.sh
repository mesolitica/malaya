WANDB_PROJECT="translation-t5-small-noisy" \
torchrun \
--nproc_per_node 4 \
-m run_t5 \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
--num_train_epochs 3 \
--eval_steps 1000000000 \
--logging_steps 10 \
--save_steps 500 \
--save_total_limit 3 \
--do_train \
--train_file mosaic-noisy-translation \
--output_dir finetune-t5-small-standard-bahasa-cased-noisy \
--per_device_train_batch_size=38 \
--per_device_eval_batch_size=4 \
--max_source_length 1536 \
--max_target_length 1536 \
--learning_rate 2e-4 \
--gradient_checkpointing true \
--bf16