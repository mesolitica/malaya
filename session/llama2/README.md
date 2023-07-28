# Finetune Llama2

## prerequisites 

1. Install libraries,

```bash
pip3 install -r requirements.txt
```

2. When you installed PyTorch with CUDA, it bundled with specific CUDA version, so I suggest to use that CUDA for `bitsandbytes`, for an example,

```python
import os
os.environ['LD_LIBRARY_PATH'] = '/home/husein/.local/lib/python3.8/site-packages/torch/lib'

import bitsandbytes
```

But before that, do not forget to symlink because `bitsandbytes` required `libcudart.so`,

```bash
ln -s ~/.local/lib/python3.8/site-packages/torch/lib/libcudart-d0da41ae.so.11.0 ~/.local/lib/python3.8/site-packages/torch/lib/libcudart.so
```

3. Prepare dataset, https://github.com/huseinzol05/malaysian-dataset/tree/master/llm/instruction,

- Malaysia Parliament QA instructions.
- Local Twitter Sentiment instructions.
- Translation instructions.
- Malay News instructions.
- Language Model on malay texts.

## how-to

### 7B

#### 512 context length

```bash
WANDB_DISABLED=true python3 main-lora.py \
--logging_steps 1 \
--dataset_name "/home/ubuntu/server2/dev/malaya/session/llama2/shuf-combine-512.jsonl" \
--max_steps 100000 \
--bf16 --bnb_4bit_compute_dtype bfloat16 \
--per_device_train_batch_size 24 \
--save_steps 500
```

#### 1024 context lnegth

```
LD_LIBRARY_PATH=/home/husein/.local/lib/python3.8/site-packages/torch/lib \
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=1 python3 main-lora.py \
--logging_steps 1 \
--dataset_name "/home/husein/dev/malaya/session/llama2/shuf-combine-1024.jsonl" \
--max_steps 100000 \
--bf16 --bnb_4bit_compute_dtype bfloat16 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--save_steps 500 \
--max_seq_length 1024 \
--output_dir "./results-1024"
```