# HuggingFace T5

## how-to

1. Run prepare dataset, [prepare-tatabahasa.ipynb](prepare-tatabahasa.ipynb).

2. Run training script,

Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

BASE model,
```
CUDA_VISIBLE_DEVICES=1 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
--num_train_epochs 5 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train-austronesian.json \
--validation_file shuffled-test-austronesian.json \
--output_dir finetune-t5-base-standard-bahasa-cased-austronesian \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 512 \
--max_target_length 512 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

SMALL model,
```
CUDA_VISIBLE_DEVICES=1 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
--num_train_epochs 5 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train-austronesian.json \
--validation_file shuffled-test-austronesian.json \
--output_dir finetune-t5-small-standard-bahasa-cased-austronesian \
--per_device_train_batch_size=24 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 512 \
--max_target_length 512 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

TINY model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-tiny-standard-bahasa-cased \
--num_train_epochs 5 \
--logging_steps 100 \
--eval_steps 10000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train-austronesian.json \
--validation_file shuffled-test-austronesian.json \
--output_dir finetune-t5-tiny-standard-bahasa-cased-austronesian \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 512 \
--max_target_length 512 \
--gradient_checkpointing true
```

## download

1. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-base-bahasa-cased
2. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-small-bahasa-cased
3. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-tiny-bahasa-cased
4. https://huggingface.co/mesolitica/finetune-translation-t5-base-standard-bahasa-cased
5. https://huggingface.co/mesolitica/finetune-translation-t5-small-standard-bahasa-cased
6. https://huggingface.co/mesolitica/finetune-translation-t5-tiny-standard-bahasa-cased