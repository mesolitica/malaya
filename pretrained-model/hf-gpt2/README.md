# GPT2 Causal LM

Original script at https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

Train GPT2 for text scoring and neural beam scoring.

## how-to

1. Train model,

```bash
WANDB_DISABLED=true \
python3 run_clm.py \
--model_name_or_path mesolitica/gpt2-117m-bahasa-cased \
--train_file train-v2.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 1000 \
--save_steps 10000 \
--evaluation_strategy steps \
--do_train \
--do_eval \
--output_dir gpt2-117m-bahasa-cased-clm \
--save_total_limit 5 \
--ignore_data_skip \
--block_size 1024 \
--fp16
```