# HuggingFace T5

## how-to

1. Run prepare dataset, [prepare-tatabahasa.ipynb](prepare-tatabahasa.ipynb).

2. Run training script,

Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

BASE model,
```
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-base-standard-bahasa-cased \
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
--train_file shuffled-train-austronesian.json \
--validation_file shuffled-test-austronesian-10k.json \
--output_dir finetune-t5-base-standard-bahasa-cased-austronesian \
--per_device_train_batch_size=24 \
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
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
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
--train_file shuffled-train-austronesian.json \
--validation_file shuffled-test-austronesian-10k.json \
--output_dir finetune-t5-small-standard-bahasa-cased-austronesian \
--per_device_train_batch_size=32 \
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
--eval_steps 100000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train-austronesian.json \
--validation_file shuffled-test-austronesian-10k.json \
--output_dir finetune-t5-tiny-standard-bahasa-cased-austronesian \
--per_device_train_batch_size=64 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 256 \
--max_target_length 256 \
--fp16
```

## download

1. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-base-bahasa-cased
2. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-small-bahasa-cased
3. https://huggingface.co/mesolitica/finetune-noisy-translation-t5-tiny-bahasa-cased
4. https://huggingface.co/mesolitica/finetune-translation-t5-base-standard-bahasa-cased
5. https://huggingface.co/mesolitica/finetune-translation-t5-small-standard-bahasa-cased
6. https://huggingface.co/mesolitica/finetune-translation-t5-tiny-standard-bahasa-cased