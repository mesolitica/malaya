Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

SUPER TINY model,
```
WANDB_DISABLED=true \
python3 run_t5_classification.py \
--model_name_or_path mesolitica/t5-super-tiny-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 5 \
--do_train \
--do_eval \
--train_file shuffled-train.json \
--validation_file test-1k.json \
--output_dir finetune-t5-super-tiny-standard-bahasa-cased \
--per_device_train_batch_size=42 \
--per_device_eval_batch_size=4 \
--max_seq_length 256 \
--ignore_data_skip \
--fp16
```

TINY model,
```
WANDB_DISABLED=true \
python3 run_t5_classification.py \
--model_name_or_path mesolitica/t5-tiny-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 5 \
--do_train \
--do_eval \
--train_file shuffled-train.json \
--validation_file test-1k.json \
--output_dir finetune-t5-tiny-standard-bahasa-cased \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=4 \
--max_seq_length 256 \
--ignore_data_skip \
--fp16
```

SMALL model,
```
WANDB_DISABLED=true \
python3 run_t5_classification.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 5 \
--do_train \
--do_eval \
--train_file shuffled-train.json \
--validation_file test-1k.json \
--output_dir finetune-t5-small-standard-bahasa-cased \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=4 \
--max_seq_length 256 \
--ignore_data_skip \
--fp16
```

BASE model,
```
WANDB_DISABLED=true \
python3 run_t5_classification.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 1000000000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 5 \
--do_train \
--train_file shuffled-train.json \
--validation_file test-1k.json \
--output_dir finetune-t5-base-standard-bahasa-cased \
--per_device_train_batch_size=24 \
--per_device_eval_batch_size=4 \
--max_seq_length 256 \
--ignore_data_skip \
--fp16
```