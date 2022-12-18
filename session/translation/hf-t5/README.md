# HuggingFace T5

## how-to

1. Run prepare dataset, [prepare-data.ipynb](prepare-data.ipynb).

2. Run training script,

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

SUPER TINY model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-super-tiny-standard-bahasa-cased \
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
--output_dir finetune-t5-super-tiny-standard-bahasa-cased \
--per_device_train_batch_size=92 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```

SUPER SUPER TINY model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-super-super-tiny-standard-bahasa-cased \
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
--output_dir finetune-t5-super-super-tiny-standard-bahasa-cased \
--per_device_train_batch_size=108 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```

SMALL FLAN model,
```
CUDA_VISIBLE_DEVICES=1 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path google/flan-t5-small \
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
--train_file shuffled-train.json \
--validation_file test.json \
--output_dir finetune-flan-t5-small \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--learning_rate 5e-5
```

BASE FLAN model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path google/flan-t5-base \
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
--train_file shuffled-train.json \
--validation_file test.json \
--output_dir finetune-flan-t5-base \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--learning_rate 5e-5
```

## download

1. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-base-bahasa-cased
2. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-small-bahasa-cased
3. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-tiny-bahasa-cased
4. https://huggingface.co/mesolitica/finetune-translation-t5-base-standard-bahasa-cased
5. https://huggingface.co/mesolitica/finetune-translation-t5-small-standard-bahasa-cased
6. https://huggingface.co/mesolitica/finetune-translation-t5-tiny-standard-bahasa-cased