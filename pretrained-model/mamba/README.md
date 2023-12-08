# Pretrain Mamba

## prerequisites 

1. Install libraries,

```bash
pip3 install mamba-ssm
```

### 1.4B, 4096 Context length

- Dataset gathered at https://github.com/malaysia-ai/dedup-text-dataset/tree/main/pretrain-llm
- We use Ray cluster to train on 5 nodes of 4x A100 80GB, https://github.com/malaysia-ai/jupyter-gpu/tree/main/ray

```bash
bash run-1b.sh
```

**Training script already hardcoded deepspeed Zero 3 config and other configs, it might only suitable to pretrain mistral from scratch**.

https://wandb.ai/mesolitica/pretrain-mamba-1.4b?workspace=user-husein-mesolitica