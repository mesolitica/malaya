# HuggingFace T5

**This directory is very lack of comments, able to understand PyTorch and HuggingFace Transformers are really helpful**.

## how-to

1. Prepare dataset, [prepare-dataset.ipynb](prepare-dataset.ipynb).

2. Run training script,

Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

SMALL model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 20 \
--eval_steps 2000 \
--save_steps 2000 \
--evaluation_strategy steps \
--save_total_limit 5 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test.json \
--output_dir finetune-t5-small-standard-bahasa-cased \
--per_device_train_batch_size=24 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--fp16 \
--gradient_checkpointing true
```

BASE model,
```
WANDB_DISABLED=true \
accelerate launch \
run_t5.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
--num_train_epochs 20 \
--logging_steps 20 \
--eval_steps 2000 \
--save_steps 2000 \
--evaluation_strategy steps \
--save_total_limit 2 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test.json \
--output_dir finetune-t5-base-standard-bahasa-cased \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--learning_rate 2e-4 \
--fp16 \
--gradient_checkpointing true
```