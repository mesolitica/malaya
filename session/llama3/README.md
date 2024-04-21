# Finetune Gemma

### Flash Attention 2

1. Install dependencies,

```bash
pip3 install flash-attn --no-build-isolation -U
pip3 install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

## Full Parameter Finetuning

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup/tree/main/gemma

### 8B, 8192 Context length

```bash
bash train-7b.sh
```

https://wandb.ai/huseinzol05/finetune-gemma-7b?workspace=user-huseinzol05