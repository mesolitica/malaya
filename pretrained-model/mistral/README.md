# Pretrain Mistral

## prerequisites 

1. Install libraries,

```bash
pip3 install -r requirements.txt
```

### Flash Attention 2

1. Install dependencies,

```bash
pip3 install flash-attn --no-build-isolation -U
pip3 install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

### 1.1B, 4096 Context length

- Dataset gathered at https://github.com/malaysia-ai/dedup-text-dataset/tree/main/pretrain-llm
- We use Ray cluster to train on 5 nodes of 4x A100 80GB, https://github.com/malaysia-ai/jupyter-gpu/tree/main/ray

```bash
bash run-1b.sh
```

**Training script already hardcoded deepspeed Zero 3 config and other configs, it might only suitable to pretrain mistral from scratch**.

https://wandb.ai/mesolitica/pretrain-mistral-1.1b?workspace=user-husein-mesolitica

### 3B, 4096 Context length

- Dataset gathered at https://github.com/malaysia-ai/dedup-text-dataset/tree/main/pretrain-llm
- We use Ray cluster to train on 5 nodes of 4x A100 80GB, https://github.com/malaysia-ai/jupyter-gpu/tree/main/ray

```bash
bash run-3b.sh
```

**Training script already hardcoded deepspeed Zero 3 config and other configs, it might only suitable to pretrain mistral from scratch**.

https://wandb.ai/mesolitica/pretrain-mistral-3b?workspace=user-husein-mesolitica

### 5B, 4096 Context length

- Dataset gathered at https://github.com/malaysia-ai/dedup-text-dataset/tree/main/pretrain-llm
- We use Ray cluster to train on 5 nodes of 8x A100 80GB, https://github.com/malaysia-ai/jupyter-gpu/tree/main/ray

```bash
bash run-5b.sh
```

**Training script already hardcoded deepspeed Zero 3 config and other configs, it might only suitable to pretrain mistral from scratch**.

https://wandb.ai/mesolitica/pretrain-mistral-5b?workspace=user-husein-mesolitica