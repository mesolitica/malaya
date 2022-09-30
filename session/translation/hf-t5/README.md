Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

SMALL model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 10 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file train.json \
--validation_file test.json \
--output_dir finetune-t5-small-standard-bahasa-cased \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```

TINY model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-tiny-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 10 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file train.json \
--validation_file test.json \
--output_dir finetune-t5-tiny-standard-bahasa-cased \
--per_device_train_batch_size=72 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```

BASE model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 10 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file train.json \
--validation_file test.json \
--output_dir finetune-t5-base-standard-bahasa-cased \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```

Using adafactor optimizer,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 10 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file train.json \
--validation_file test.json \
--output_dir finetune-t5-small-standard-bahasa-cased \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=4 \
--overwrite_output_dir \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16 \
--optim adafactor \
--learning_rate 1e-3 \
--lr_scheduler_type constant
```