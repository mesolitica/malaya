# Finetune Llama-3

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
bash train-8b.sh
```

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

## Unsloth

1. Install dependencies,

```
pip3 install pip -U
pip3 uninstall torch torchvision flash-attn -y
pip3 install torch torchvision
pip3 install mosaicml-streaming
pip3 install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip3 install flash-attn --no-build-isolation
```