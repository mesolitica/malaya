# Finetune Mistral

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

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup#mistral

### 7B, 4096 Context length

```bash
bash run.sh
```

https://wandb.ai/mesolitica/fpf-mistral-7b-hf?workspace=user-husein-mesolitica

### 7B, 32768 Context length

```bash
bash run-32k.sh
```

https://wandb.ai/mesolitica/fpf-mistral-7b-hf-32k?workspace=user-husein-mesolitica

### 1B, 32768 Context length

1. Run [mistral-1b.ipynb](mistral-1b.ipynb).

2. Run training,

```bash
bash run-32k-1b.sh
```

https://wandb.ai/mesolitica/fpf-mistral-1b-hf-32k?workspace=user-husein-mesolitica

### Instructions, 7B, 16384 Context length

```
bash run-instructions-16k.sh
```

https://wandb.ai/mesolitica/fpf-mistral-7b-hf-instructions-16k?workspace=user-husein-mesolitica

### MaLLaM 1.1B

```bash
bash run-instructions-20k-mallam-1.1b.sh
```

### MaLLaM 3B

```bash
bash run-instructions-20k-mallam-3b.sh
```

### MaLLaM 5B

```bash
bash run-instructions-20k-mallam-5b.sh
```