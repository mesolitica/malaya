WANDB_PROJECT="nanot5-base-malaysian-cased-translation-v2-coding" \
CUDA_VISIBLE_DEVICES=0 \
python3.10 run_t5_v2.py \
--model_name_or_path mesolitica/nanot5-base-malaysian-translation-v2 \
--num_train_epochs 2 \
--eval_steps 1000000000 \
--logging_steps 2 \
--save_steps 200 \
--save_total_limit 3 \
--do_train \
--train_file mosaic-coding \
--output_dir nanot5-base-malaysian-cased-translation-v2-coding \
--dataloader_num_workers=10 \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=3 \
--gradient_accumulation_steps=8 \
--max_source_length 2048 \
--max_target_length 2048 \
--learning_rate 2e-5 \
--gradient_checkpointing true \
--weight_decay 0.01 \
--bf16