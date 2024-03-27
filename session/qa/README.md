# HuggingFace T5

**This directory is very lack of comments, able to understand PyTorch and HuggingFace Transformers are really helpful**.

## how-to

1. Prepare dataset,

- For T5 Bahasa, [prepare-data.ipynb](prepare-data.ipynb).
- For Flan T5, [prepare-data-flan.ipynb](prepare-data-flan.ipynb).

2. Run training script,

TINY model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-tiny-standard-bahasa-cased \
--num_train_epochs 50 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test-10k.json \
--output_dir finetune-t5-tiny-standard-bahasa-cased-qa \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--fp16 \
--preprocessing_num_workers 5 \
--gradient_checkpointing true
```

SMALL model,
```
CUDA_VISIBLE_DEVICES=1 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test-10k.json \
--output_dir finetune-t5-small-standard-bahasa-cased-qa \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--fp16 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

BASE model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 2 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test-10k.json \
--output_dir finetune-t5-base-standard-bahasa-cased-qa-v2 \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--fp16 \
--gradient_checkpointing true
```

## download

1. BASE Flan T5, https://huggingface.co/mesolitica/finetune-extractive-qa-flan-t5-base
2. SMALL Flan T5, https://huggingface.co/mesolitica/finetune-extractive-qa-flan-t5-small
3. TINY T5 Bahasa, https://huggingface.co/mesolitica/finetune-extractive-qa-t5-tiny-standard-bahasa-cased
4. SMALL T5 Bahasa, https://huggingface.co/mesolitica/finetune-extractive-qa-t5-small-standard-bahasa-cased
5. BASE T5 Bahasa, https://huggingface.co/mesolitica/finetune-extractive-qa-t5-base-standard-bahasa-cased