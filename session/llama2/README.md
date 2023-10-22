# Finetune Llama2

## prerequisites 

1. Install libraries,

```bash
pip3 install -r requirements.txt
```

### Flash Attention 2

1. Install dependencies,

```bash
pip3 install flash-attn --no-build-isolation
pip3 install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

## Full Parameter Finetuning

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup#llama2

### VM Spec

We use Azure Kubernetes Standard_NC96ads_A100_v4 for each FPF.

1. 96 vCPU
2. 880 GB RAM
3. 4x A100 80GB with topology,

```bash
nvidia-smi topo -m
```

```text
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV12    SYS     SYS     0-23    0
GPU1    NV12     X      SYS     SYS     24-47   1
GPU2    SYS     SYS      X      NV12    48-71   2
GPU3    SYS     SYS     NV12     X      72-95   3
```

- When you use Kubernetes, `/dev/shm` default to 64 MB, and this caused bus error for NCCL. this issue happened on AWS p5 and Azure NC A100 v4-series, so to solve it, scale it up, eg, https://github.com/malaysia-ai/jupyter-gpu/blob/main/aks/jupyter4.yaml#L99
- Always perform NCCL test, https://github.com/NVIDIA/nccl-tests,

```bash
git clone https://github.com/NVIDIA/nccl-tests && cd nccl-tests
make
./build/all_gather_perf -g 4
```

```text
# nThread 1 nGpus 4 minBytes 33554432 maxBytes 33554432 step: 1048576(bytes) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    622 on jupyter4-0 device  0 [0x00] NVIDIA A100 80GB PCIe
#  Rank  1 Group  0 Pid    622 on jupyter4-0 device  1 [0x00] NVIDIA A100 80GB PCIe
#  Rank  2 Group  0 Pid    622 on jupyter4-0 device  2 [0x00] NVIDIA A100 80GB PCIe
#  Rank  3 Group  0 Pid    622 on jupyter4-0 device  3 [0x00] NVIDIA A100 80GB PCIe
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
    33554432       2097152     float    none      -1    973.2   34.48   25.86      0    969.7   34.60   25.95      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 25.905
```

Good to go!

### 7B, 2048 Context length

```bash
WANDB_PROJECT=fpf-Llama-2-7b-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--per_device_train_batch_size 16 \
--gradient_accumulation_steps 1 \
--output_dir fpf-7b \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "combine.jsonl" \
--logging_steps 1 \
--learning_rate 5e-5 \
--block_size 2048 \
--save_steps 200 \
--save_total_limit 2 \
--gradient_checkpointing true
```

https://wandb.ai/mesolitica/fpf-Llama-2-7b-hf?workspace=user-husein-mesolitica

### 7B, 16384 Context length

```bash
WANDB_PROJECT=fpf-Llama-2-7b-16k-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--per_device_train_batch_size 6 \
--gradient_accumulation_steps 1 \
--output_dir fpf-7b-16k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 10 \
--train_file "combine.jsonl" \
--logging_steps 1 \
--learning_rate 5e-5 \
--block_size 16384 \
--save_steps 200 \
--save_total_limit 2 \
--gradient_checkpointing true \
--use_flash_attention2 true
```

https://wandb.ai/mesolitica/fpf-Llama-2-7b-16k-hf?workspace=user-husein-mesolitica

### 7B, 32768 Context length

```bash
WANDB_PROJECT=fpf-Llama-2-7b-32k-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path mesolitica/llama-7b-hf-16384-fpf \
--per_device_train_batch_size 3 \
--gradient_accumulation_steps 1 \
--output_dir fpf-7b-32k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "combine-v2.jsonl" \
--logging_steps 1 \
--learning_rate 2e-5 \
--lr_scheduler_type "linear" \
--max_grad_norm 0.5 \
--block_size 32768 \
--save_steps 100 \
--save_total_limit 2 \
--gradient_checkpointing true \
--use_flash_attention2 true
```

https://wandb.ai/mesolitica/fpf-Llama-2-7b-32k-hf?workspace=user-husein-mesolitica

### 13B, 2048 Context length

```bash
WANDB_PROJECT=fpf-Llama-2-13b-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path meta-llama/Llama-2-13b-hf \
--per_device_train_batch_size 12 \
--gradient_accumulation_steps 1 \
--output_dir fpf-13b \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "combine.jsonl" \
--logging_steps 1 \
--learning_rate 5e-5 \
--block_size 2048 \
--save_steps 200 \
--save_total_limit 2 \
--gradient_checkpointing true
```

https://wandb.ai/mesolitica/fpf-Llama-2-13b-hf?workspace=user-husein-mesolitica

### 13B, 16384 Context length

```bash
WANDB_PROJECT=fpf-Llama-2-13b-16k-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path meta-llama/Llama-2-13b-hf \
--per_device_train_batch_size 5 \
--gradient_accumulation_steps 1 \
--output_dir fpf-13b-16k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 10 \
--train_file "combine.jsonl" \
--logging_steps 1 \
--learning_rate 5e-5 \
--block_size 16384 \
--save_steps 200 \
--save_total_limit 2 \
--gradient_checkpointing true \
--use_flash_attention2 true
```

https://wandb.ai/mesolitica/fpf-Llama-2-13b-16k-hf?workspace=user-husein-mesolitica

### 13B, 32768 context length

```bash
WANDB_PROJECT=fpf-Llama-2-13b-32k-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path mesolitica/llama-13b-hf-16384-fpf \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 1 \
--output_dir fpf-13b-32k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "combine-v2.jsonl" \
--logging_steps 1 \
--learning_rate 2e-5 \
--lr_scheduler_type "linear" \
--max_grad_norm 0.5 \
--block_size 32768 \
--save_steps 50 \
--save_total_limit 2 \
--gradient_checkpointing true \
--use_flash_attention2 true
```

https://wandb.ai/mesolitica/fpf-Llama-2-13b-32k-hf?workspace=user-husein-mesolitica

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

## Full Parameter Finetuning  smaller models

### 600M, 32768 context length, flash attention 2

600M derived from first 2 layers 7B model,

```bash
WANDB_PROJECT=fpf-Llama-2-600m-32k-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path "./600m" \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--output_dir fpf-600m-32k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "combine-v2.jsonl" \
--logging_steps 1 \
--learning_rate 2e-5 \
--lr_scheduler_type "linear" \
--max_grad_norm 0.5 \
--block_size 32768 \
--save_steps 100 \
--save_total_limit 2 \
--gradient_checkpointing true \
--use_flash_attention2 true
```

### 1B, 32768 context length, flash attention 2

1B derived from first 4 layers 7B model,

```bash
WANDB_PROJECT=fpf-Llama-2-1b-32k-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path "./1b" \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--output_dir fpf-1b-32k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "combine-v2.jsonl" \
--logging_steps 1 \
--learning_rate 2e-5 \
--lr_scheduler_type "linear" \
--max_grad_norm 0.5 \
--block_size 32768 \
--save_steps 100 \
--save_total_limit 2 \
--gradient_checkpointing true \
--use_flash_attention2 true
```

### 2B, 32768 context length, flash attention 2

2B derived from first 5 layers 13B model,

```bash
WANDB_PROJECT=fpf-Llama-2-2b-32k-hf ~/.local/bin/deepspeed run_clm.py \
--deepspeed ds_config_zero3.json \
--model_name_or_path "./2b" \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--output_dir fpf-2b-32k \
--bf16 \
--do_train \
--do_eval false \
--num_train_epochs 1 \
--train_file "combine-v2.jsonl" \
--logging_steps 1 \
--learning_rate 2e-5 \
--lr_scheduler_type "linear" \
--max_grad_norm 0.5 \
--block_size 32768 \
--save_steps 100 \
--save_total_limit 2 \
--gradient_checkpointing true \
--use_flash_attention2 true
```

## LORA Instruction

Dataset prepared at https://github.com/huseinzol05/malaysian-dataset/tree/master/llm/instruction

**16384 context length trained on A100**.

### 7B, 16384 context length, flash attention 2

```bash
WANDB_PROJECT=qlora-7b-instructions-16k-improve \
python3 main-lora.py \
--model_name "mesolitica/llama-7b-hf-32768-fpf" \
--output_dir "./results-7b-16384-improve" \
--dataset_name "shuf-combine-1536-v2.jsonl" \
--max_seq_length 16384 \
--group_by_length true \
--bnb_4bit_compute_dtype bfloat16 \
--save_steps 1000 \
--save_total_limit 2 \
--logging_steps 1 \
--max_steps 100000 \
--bf16 \
--learning_rate 2e-4 \
--optim "paged_adamw_32bit" \
--lr_scheduler_type "linear" \
--warmup_ratio 0.03 \
--max_grad_norm 0.3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing true \
--use_flash_attention2 true
```

### 13B, 16384 context length, flash attention 2

**required deepspeed**.

```bash
WANDB_PROJECT=qlora-13b-instructions-16k-improve \
~/.local/bin/deepspeed main-lora.py \
--deepspeed ds_config_zero2.json \
--model_name "mesolitica/llama-13b-hf-32768-fpf" \
--output_dir "./results-13b-16384-improve" \
--dataset_name "shuf-combine-1536-v2.jsonl" \
--max_seq_length 16384 \
--group_by_length true \
--bnb_4bit_compute_dtype bfloat16 \
--save_steps 100 \
--save_total_limit 2 \
--logging_steps 1 \
--max_steps 200000 \
--bf16 \
--learning_rate 2e-4 \
--optim "paged_adamw_32bit" \
--lr_scheduler_type "linear" \
--warmup_ratio 0.03 \
--max_grad_norm 0.3 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing true \
--use_flash_attention2 true
```