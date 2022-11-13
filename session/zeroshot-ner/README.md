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
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test-10k.json \
--output_dir finetune-t5-small-standard-bahasa-cased-ner-new \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```

BASE model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
--num_train_epochs 10 \
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
--output_dir finetune-t5-base-standard-bahasa-cased-ner \
--per_device_train_batch_size=16 \
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
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test-10k.json \
--output_dir finetune-t5-tiny-standard-bahasa-cased-ner \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```