# Finetune Qwen2Audio

## how to finetune

1. Install libraries,

```bash
apt update
apt install ninja-build vim -y
pip3 install torch==2.5.1 torchaudio==2.5.1 deepspeed==0.15.4
pip3 install datasets evaluate peft librosa soundfile
pip3 install git+https://github.com/mesolitica/qwen2audio-multipack
pip3 install git+https://github.com/mesolitica/ml-cross-entropy-lora-lm-head
```

2. Start finetune

```bash
wget https://raw.githubusercontent.com/mesolitica/malaya/refs/heads/master/session/llama3/ds_config_zero3.json
bash 128.sh
```