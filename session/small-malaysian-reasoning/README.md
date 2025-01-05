# Small Malaysian Reasoning

## how to

1. Install necessary libraries,

```bash
pip3 install transformers==4.47.0 accelerate==1.1.1
pip3 install git+https://github.com/mesolitica/ml-cross-entropy-lora-lm-head
```

2. Run training,

```bash
bash 3b.sh
```

## how to flex attention

1. Install necessary libraries,

```bash
pip3 install accelerate==1.1.1
pip3 install git+https://github.com/mesolitica/llama-flex-attention-multipack
pip3 install git+https://github.com/mesolitica/ml-cross-entropy-lora-lm-head
```

2. Run training,

```bash
bash 3b-flex.sh
```