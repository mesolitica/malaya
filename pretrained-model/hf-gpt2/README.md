# GPT2 Causal LM

Original script at https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

Train GPT2 for text scoring and neural beam scoring.

## how-to

### Train 117M

1. Prepare model, [prepare-gpt2.ipynb](prepare-gpt2.ipynb).

2. Train the model,

```bash
CUDA_VISIBLE_DEVICES=1 \
WANDB_DISABLED=true \
python3 run_clm.py \
--model_name_or_path malay-cased-gpt2 \
--train_file combined-gpt2.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 8 \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 1000 \
--save_steps 1000 \
--evaluation_strategy steps \
--do_train \
--do_eval \
--output_dir malay-cased-gpt2-clm-v2 \
--save_total_limit 5 \
--ignore_data_skip \
--block_size 512 \
--fp16
```

### Train 355M

1. Prepare model, [prepare-gpt2-355m.ipynb](prepare-gpt2-355m.ipynb).

2. Train the model,

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_DISABLED=true \
python3 run_clm.py \
--model_name_or_path malay-cased-gpt2-355m \
--train_file combined-gpt2.txt \
--validation_file sample-wiki.txt \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 14 \
--num_train_epochs 10 \
--logging_steps 100 \
--eval_steps 1000 \
--save_steps 1000 \
--evaluation_strategy steps \
--do_train \
--do_eval \
--output_dir malay-cased-gpt2-355m-clm \
--save_total_limit 3 \
--ignore_data_skip \
--block_size 512 \
--fp16 \
--gradient_checkpointing true
```