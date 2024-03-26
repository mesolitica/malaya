# Reranker

## how-to

```bash
python3 run.py \
--output_dir="./embedding-model-llama-600m" \
--model_name_or_path="mesolitica/llama-600m-hf-32768-fpf" \
--train_data="shuf-train-embedding.jsonl" \
--per_device_train_batch_size="8" \
--learning_rate="2e-5" \
--num_train_epochs="1" \
--max_seq_length 1024 \
--save_steps="300" \
--save_total_limit="3" \
--do_train \
--gradient_checkpointing \
--logging_steps 1 \
--query_max_len 1024 \
--passage_max_len 1024 \
--train_group_size 3  \
--max_grad_norm 1.0 \
--bf16
```