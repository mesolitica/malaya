# PEFT LoRa

## prerequisites 

1. Install libraries,

```bash
pip3 install bitsandbytes==0.37.2
pip3 install git+https://github.com/huggingface/peft
CUDA_HOME=/usr/local/cuda-11.2 pip3 install deepspeed
```

1. When you installed PyTorch with CUDA, it bundled with specific CUDA version, so I suggest to use that CUDA for `bitsandbytes`, for an example,

```python
import os
os.environ['LD_LIBRARY_PATH'] = '/home/husein/.local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib'

import bitsandbytes
```

But before that, do not forget to symlink because `bitsandbytes==0.37.2` required `libcudart.so`,

```bash
ln -s ~/.local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib/libcudart.so.11.0 ~/.local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib/libcudart.so
```

## Train GPTJ

GPT6J is not sharded, so when you loaded GPTJ, at first it will consumed a lot of memory, so beware!

```bash
CUDA_VISIBLE_DEVICES='0' python3 gptj6b.py --output_dir='./lora-gptj-v5' \
--lora_r=16 \
--lora_dropout=0.05 \
--cutoff_len=1536 \
--data_files='shuf-combined-v3.jsonl'
```

## Train GPT-NeoX

### LoRA

```bash
CUDA_VISIBLE_DEVICES='1' python3 gptneox.py --output_dir='./lora-gptneox-v5' --save_steps=100 --lora_r=16 --lora_dropout=0.05 --cutoff_len=1536 --data_files='shuf-combined-v3.jsonl' \
--lora_target_modules='[query_key_value, xxx]' --base_model='EleutherAI/pythia-2.8b'
```

```bash
CUDA_VISIBLE_DEVICES='1' python3 gptneox.py --output_dir='./lora-gptneox-v7' --save_steps=100 --lora_r=16 --lora_dropout=0.05 --cutoff_len=1536 --data_files='shuf-combined-v4.jsonl' \
--lora_target_modules='[query_key_value, xxx]' --base_model='EleutherAI/pythia-6.9b'
```

### Without LoRA

```bash
CUDA_VISIBLE_DEVICES='0' python3 gptneox-notlora.py --output_dir='./gptneox-pythia-1b' \
--save_steps=1000 --data_files='shuf-combined-v4.jsonl' --base_model='EleutherAI/pythia-1b' \
--resume_from_checkpoint=True
```

```bash
CUDA_VISIBLE_DEVICES='1' python3 gptneox-notlora.py --output_dir='./gptneox-pythia-410m' \
--save_steps=200 --data_files='shuf-combined-v4.jsonl' --base_model='EleutherAI/pythia-410m' \
--batch_size=8
```

### Without LoRA and Accelerate

```bash
accelerate config
```

```bash
CUDA_VISIBLE_DEVICES='1' accelerate launch gptneox-notlora-notrainer.py --output_dir='./gptneox-pythia-1.4b' \
--save_steps=200 --data_files='combined.jsonl' --base_model='EleutherAI/pythia-1.4b' \
--gradient_accumulation_steps=1
```
