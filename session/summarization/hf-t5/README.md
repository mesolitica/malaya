# HuggingFace T5

**This directory is very lack of comments, able to understand PyTorch and HuggingFace Transformers are really helpful**.

## how-to

1. Prepare dataset,

- For mixed language, [prepare-dataset.ipynb](prepare-dataset.ipynb).
- For MS language only, [prepare-dataset-ms.ipynb](prepare-dataset-ms.ipynb).

2. Run training script,


Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

SMALL model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
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
--train_file shuffled-train.json \
--validation_file shuffled-test.json \
--output_dir finetune-t5-small-standard-bahasa-cased \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--fp16 \
--gradient_checkpointing true
```

SMALL MS model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
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
--train_file shuffled-train-ms.json \
--validation_file shuffled-test-ms.json \
--output_dir finetune-t5-small-standard-bahasa-cased-ms \
--per_device_train_batch_size=32 \
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
CUDA_VISIBLE_DEVICES=1 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
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
--train_file shuffled-train.json \
--validation_file shuffled-test.json \
--output_dir finetune-t5-base-standard-bahasa-cased \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--fp16 \
--gradient_checkpointing true
```

BASE MS model,
```
CUDA_VISIBLE_DEVICES=1 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
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
--train_file shuffled-train-ms.json \
--validation_file shuffled-test-ms.json \
--output_dir finetune-t5-base-standard-bahasa-cased-ms \
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

1. https://huggingface.co/mesolitica/finetune-summarization-t5-base-standard-bahasa-cased
2. https://huggingface.co/mesolitica/finetune-summarization-t5-small-standard-bahasa-cased