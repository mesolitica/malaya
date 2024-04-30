# Finetune Gemma

## Flash Attention 2

1. Install dependencies,

```bash
pip3 install flash-attn --no-build-isolation -U
pip3 install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary
```

## Continue Pretraining

Dataset prepared at https://github.com/malaysia-ai/text-dataset-dedup/tree/main/gemma

### 8B, 8192 Context length

```bash
bash train-7b.sh
```

https://wandb.ai/huseinzol05/finetune-gemma-7b?workspace=user-huseinzol05

## Full Parameter Finetuning

```bash
bash run-instructions-16k.sh
```

## Extend 1M context length

1. git clone https://github.com/jzhang38/EasyContext

```bash
git clone https://github.com/jzhang38/EasyContext
```

2. Install dependencies,

```bash
cd EasyContext
pip install --pre torch==2.4.0.dev20240324  --index-url https://download.pytorch.org/whl/nightly/cu118
pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt
```