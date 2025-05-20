# MalayMMLU 0-Shot

Evaluate MalayMMLU 0-Shot using vLLM Guided Decoding on first token predict with 5 votes.

## Requirements

**We recommend to use virtual environment.**

```bash
python3 -m venv vllm
vllm/bin/pip3 install vllm pandas click
```

## how to evaluate

```bash
vllm/bin/python3 evaluate_vllm.py \
--model "mesolitica/Malaysian-Qwen2.5-32B-Instruct"
```