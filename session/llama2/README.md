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

3. Install apex with specific options,

```bash
git clone https://github.com/NVIDIA/apex && cd apex
python3 setup.py install --user --cpp_ext --cuda_ext
```

## Full Parameter Finetuning

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup#merge-and-prepare-to-huggingface-dataset

### VM Spec

We use Azure Kubernetes Standard_NC96ads_A100_v4 for each FPF.

1. 96 vCPU
2. 880 GB RAM
3. 4x A100 with topology,

```text
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV12    SYS     SYS     0-23    0
GPU1    NV12     X      SYS     SYS     24-47   1
GPU2    SYS     SYS      X      NV12    48-71   2
GPU3    SYS     SYS     NV12     X      72-95   3
```

### 7B, 1024 Context length

```bash
~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--output_dir fpf-7b \
--fp16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "1m-combine.json" \
--validation_file "1k-fix-combine.json" \
--logging_steps 1 \
--learning_rate 5e-5 \
--block_size 1024 \
--save_steps 1000 \
--save_total_limit 3 \
--streaming
```

https://wandb.ai/mesolitica/huggingface?workspace=user-husein-mesolitica

### 13B, 1024 Context length

```bash
WANDB_PROJECT=fpf-Llama-2-13b-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path meta-llama/Llama-2-13b-hf \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--output_dir fpf-13b \
--fp16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "1m-combine.json" \
--validation_file "1k-fix-combine.json" \
--logging_steps 1 \
--learning_rate 5e-5 \
--block_size 1024 \
--save_steps 1000 \
--save_total_limit 3 \
--streaming
```

## Check memory usage DeepSpeed 3

### 7B

```bash
python3 -c 'from transformers import AutoModelForCausalLM; \
import torch; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)'
```

```text
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 4 GPUs per node.
SW: Model with 6738M total params, 131M largest layer params.
  per CPU  |  per GPU |   Options
  169.44GB |   0.49GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  169.44GB |   0.49GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
  150.62GB |   3.63GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  150.62GB |   3.63GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    2.93GB |  28.73GB | offload_param=none, offload_optimizer=none, zero_init=1
  150.62GB |  28.73GB | offload_param=none, offload_optimizer=none, zero_init=0
```

### 13B

```bash
python3 -c 'from transformers import AutoModelForCausalLM; \
import torch; \
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", torch_dtype=torch.bfloat16); \
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)'
```

```text
SW: Model with 13015M total params, 163M largest layer params.
  per CPU  |  per GPU |   Options
  327.29GB |   0.61GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
  327.29GB |   0.61GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
  290.93GB |   6.67GB | offload_param=none, offload_optimizer=cpu , zero_init=1
  290.93GB |   6.67GB | offload_param=none, offload_optimizer=cpu , zero_init=0
    3.66GB |  55.16GB | offload_param=none, offload_optimizer=none, zero_init=1
  290.93GB |  55.16GB | offload_param=none, offload_optimizer=none, zero_init=0
```

## LORA Instruction

Dataset prepared at https://github.com/huseinzol05/malaysian-dataset/tree/master/llm/instruction

### 7B, 1536 context length

```bash
LD_LIBRARY_PATH=/home/husein/.local/lib/python3.8/site-packages/torch/lib \
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 python3 main-lora.py \
--logging_steps 1 \
--dataset_name "/home/husein/dev/malaya/session/llama2/shuf-combine-1536.jsonl" \
--max_steps 500000 \
--bf16 --bnb_4bit_compute_dtype bfloat16 \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 2 \
--save_steps 500 \
--max_seq_length 1536 \
--output_dir "./results-1536-v2"
```

### 7B, 1536 context length, packing

```bash
LD_LIBRARY_PATH=/home/husein/.local/lib/python3.8/site-packages/torch/lib \
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=1 python3 main-lora-packing.py \
--logging_steps 1 \
--dataset_name "/home/husein/dev/malaya/session/llama2/shuf-combine-1536.jsonl" \
--max_steps 500000 \
--bf16 --bnb_4bit_compute_dtype bfloat16 \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 2 \
--save_steps 500 \
--max_seq_length 1536 \
--output_dir "./results-1024-packing"
```