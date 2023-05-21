# HuggingFace T5

## how-to T5

1. Run prepare dataset, [prepare-data.ipynb](prepare-data.ipynb).

2. Run training script,

Original script, https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

BASE model,
```
CUDA_VISIBLE_DEVICES='0' \
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
--train_file shuffled-train.json \
--validation_file test.json \
--output_dir finetune-t5-base-standard-bahasa-cased \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--max_source_length 1536 \
--max_target_length 1536 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

SMALL model,
```
CUDA_VISIBLE_DEVICES='1' \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-small-standard-bahasa-cased \
--num_train_epochs 3 \
--logging_steps 100 \
--eval_steps 1000000000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test.json \
--output_dir finetune-t5-small-standard-bahasa-cased \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--max_source_length 1536 \
--max_target_length 1536 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

TINY model,
```
CUDA_VISIBLE_DEVICES='0' \
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path mesolitica/t5-tiny-standard-bahasa-cased \
--num_train_epochs 3 \
--logging_steps 100 \
--eval_steps 1000000000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--source_lang src \
--target_lang tgt \
--train_file shuffled-train.json \
--validation_file test.json \
--output_dir finetune-t5-tiny-standard-bahasa-cased \
--per_device_train_batch_size=12 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--max_source_length 1536 \
--max_target_length 1536 \
--learning_rate 5e-5 \
--gradient_checkpointing true \
--ignore_data_skip False
```

## how-to causal

1. Run prepare dataset, [prepare-data.ipynb](prepare-data.ipynb).

2. Run training script,

```bash
CUDA_VISIBLE_DEVICES='0' python3 causal.py --output_dir='./gptneox-pythia-160m' \
--save_steps=1000 --data_files='test.json' --base_model='EleutherAI/pythia-160m' \
--resume_from_checkpoint=True --learning_rate=2e-5
```