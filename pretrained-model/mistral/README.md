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

## Pretrain

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup#pretrain

### 191M, 4096 Context length

```bash
bash run-191M.sh
```

https://wandb.ai/mesolitica/mistral-158M?workspace=user-husein-mesolitica

### 349M, 4096 Context length

```bash
bash run-349M.sh
```

https://wandb.ai/mesolitica/mistral-349M?workspace=user-husein-mesolitica

### 1.1B, 4096 Context length

- Dataset gathered at https://github.com/malaysia-ai/dedup-text-dataset/tree/main/pretrain-llm
- We use Ray cluster to train on 5 nodes of 4x A100 80GB, https://github.com/malaysia-ai/jupyter-gpu/tree/main/ray

```bash
bash run-1b.sh
```

### 3B, 4096 Context length

- Dataset gathered at https://github.com/malaysia-ai/dedup-text-dataset/tree/main/pretrain-llm
- We use Ray cluster to train on 5 nodes of 4x A100 80GB, https://github.com/malaysia-ai/jupyter-gpu/tree/main/ray

```bash
bash run-3b.sh
```

https://wandb.ai/mesolitica/pretrain-mistral-3b?workspace=user-husein-mesolitica