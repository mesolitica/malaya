# Small Malaysian Reasoning

## how to train

### SFT

1. Install necessary libraries,

```bash
pip3 uninstall torch torchvision torchaudio -y
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
pip3 install mosaicml-streaming transformers==4.47.0 accelerate==1.1.1
pip3 install git+https://github.com/mesolitica/ml-cross-entropy-lora-lm-head
pip3 install trl==0.14.0 vllm
```

2. Download dataset,

```bash
pip3 install huggingface-hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='huseinzol05/llama3.2-language-multipack-3k', repo_type='dataset', local_dir = './packing-3k')
"
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='huseinzol05/llama3.2-reasoning-multipack-3k', repo_type='dataset', local_dir = './packing-3k-reasoning')
"
```

3. Run training,

```bash
bash 3b.sh
```

### GRPO

1. Install necessary libraries,

```bash
pip3 install datasets transformers==4.47.0 accelerate==1.1.1 trl==0.14.0 vllm peft wandb
```

2. Run training,

```bash
bash 3b-reasoning.sh
```