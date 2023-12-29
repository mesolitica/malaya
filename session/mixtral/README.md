# Finetune Mixtral

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

## QLORA

### 16384 Context length

```bash
bash mixtral-lora-16k-deepspeed.sh
```