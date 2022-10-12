# Long-T5-Bahasa

Thanks to Google for opensourcing most of the source code to develop Long T5, https://github.com/google-research/longt5, and HuggingFace translated to PyTorch, https://huggingface.co/docs/transformers/model_doc/longt5

**This directory is very lack of comments, understand Tensorflow, Tensorflow estimator and Tensorflow Dataset are really helpful**.

## Objective

1. Provide T5 for Bahasa.

## Acknowledgement

Thanks to [Mesolitica](https://mesolitica.com/) for sponsoring GPU clouds to train Long T5 for Bahasa.

## How-to

TGLOBAL BASE model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path ./ms-long-t5-tglobal-base \
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
--train_file train-longer.json \
--validation_file test-longer.json \
--output_dir translation-long-t5-tglobal-base \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--warmup_steps 100000 \
--weight_decay 0.1 \
--gradient_checkpointing true
```

TGLOBAL BASE model,
```
WANDB_DISABLED=true \
python3 run_t5.py \
--model_name_or_path ./ms-long-t5-local-base \
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
--train_file train-longer.json \
--validation_file test-longer.json \
--output_dir translation-long-t5-local-base-v2 \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--ignore_data_skip \
--max_source_length 1024 \
--max_target_length 1024 \
--warmup_steps 100000 \
--weight_decay 0.1 \
--gradient_checkpointing true
```