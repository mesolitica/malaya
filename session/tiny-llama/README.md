# Finetune TinyLlama

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

## Full Parameter Finetuning

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup/tree/main/tinyllama

### 4096 Context length

```bash
bash train.sh
```

https://wandb.ai/mesolitica/finetune-tinyllama-1.1B?workspace=user-husein-mesolitica

### Instructions, 7B, 16384 Context length

```
bash run-instructions-16k.sh
```

https://wandb.ai/mesolitica/fpf-mistral-7b-hf-instructions-16k?workspace=user-husein-mesolitica