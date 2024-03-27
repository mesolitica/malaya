# HuggingFace T5

## how-to

1. Run prepare dataset, [prepare-data.ipynb](prepare-data.ipynb).

2. Run training script,

Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

BASE model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/finetune-translation-t5-base-standard-bahasa-cased-v2 \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 100000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file shuffled-test.json \
--output_dir finetune-t5-base-standard-bahasa-cased-combined \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 512 \
--max_target_length 512 \
--fp16 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

SMALL model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/finetune-translation-t5-small-standard-bahasa-cased-v2 \
--num_train_epochs 5 \
--logging_steps 100 \
--eval_steps 100000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file shuffled-test.json \
--output_dir finetune-t5-small-standard-bahasa-cased-combined \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 512 \
--max_target_length 512 \
--fp16 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

TINY model,
```
CUDA_VISIBLE_DEVICES=1 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/finetune-translation-t5-tiny-standard-bahasa-cased-v2 \
--num_train_epochs 5 \
--logging_steps 100 \
--eval_steps 100000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file shuffled-test.json \
--output_dir finetune-t5-tiny-standard-bahasa-cased-combined \
--per_device_train_batch_size=42 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 512 \
--max_target_length 512 \
--fp16 \
--gradient_checkpointing true
```