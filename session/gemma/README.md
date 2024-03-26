# Finetune Gemma

### Flash Attention 2

1. Install dependencies,

```bash
pip3 install flash-attn --no-build-isolation -U
pip3 install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

## Full Parameter Finetuning

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup/tree/main/gemma

### 2B, 8192 Context length

```bash
bash train-2b.sh
```

https://wandb.ai/huseinzol05/finetune-gemma-2b?workspace=user-huseinzol05

### 7B, 8192 Context length

```bash
bash train-7b.sh
```

https://wandb.ai/huseinzol05/finetune-gemma-7b?workspace=user-huseinzol05

### Instructions, 2B, 16384 Context length

```
bash run-instructions-16k.sh
```

https://wandb.ai/huseinzol05/gemma-2B-8192-fpf-instructions-16k?workspace=user-huseinzol05