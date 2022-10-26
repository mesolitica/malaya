Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

```
shuf train-noisy.json > train-noisy-shuffled.json
shuf test-noisy.json > test-noisy-shuffled.json
```

BASE model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/finetune-translation-t5-base-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 5 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file train-noisy-shuffled.json \
--validation_file test-noisy-shuffled.json \
--output_dir finetune-t5-base-noisy-bahasa-cased \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```

SMALL model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/finetune-translation-t5-small-standard-bahasa-cased \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--preprocessing_num_workers 10 \
--evaluation_strategy steps \
--save_total_limit 5 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file train-noisy-shuffled.json \
--validation_file test-noisy-shuffled.json \
--output_dir finetune-t5-small-noisy-bahasa-cased \
--per_device_train_batch_size=54 \
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
--model_name_or_path mesolitica/t5-tiny-bahasa-cased \
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
--train_file train-noisy.json \
--validation_file test-noisy.json \
--output_dir finetune-t5-tiny-noisy-bahasa-cased \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```