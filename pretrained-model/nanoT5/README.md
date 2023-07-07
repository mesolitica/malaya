# nanoT5

Originally from https://github.com/PiotrNawrot/nanoT5

## how-to

1. Prepare tokenizer and model,

- BASE, [prepare-tokenizer-base-model.ipynb](prepare-tokenizer-base-model.ipynb).
- SMALL, [prepare-tokenizer-small-model.ipynb](prepare-tokenizer-small-model.ipynb).

2. Prepare dataset [prepare-dataset.ipynb](prepare-dataset.ipynb).

3. Train,

- BASE,

```bash
python3 train.py \
optim.name=adamwscale \
optim.batch_size=128 \
optim.lr_scheduler=cosine \
optim.grad_acc=4 \
model.name="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/out-base-1.1" \
model.random_init=false \
data.filename.train="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/combine.jsonl" \
data.filename.test="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/combine.jsonl" \
logging.every_steps=1 \
checkpoint.every_steps=10000 \
optim.total_steps=50000000 \
model.klass="hf_t5"
```

- SMALL,

```bash
python3 train.py \
optim.name=adamwscale \
optim.batch_size=128 \
optim.lr_scheduler=cosine \
optim.grad_acc=4 \
model.name="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/out-small-1.1" \
model.random_init=false \
data.filename.train="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/combine.jsonl" \
data.filename.test="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/combine.jsonl" \
logging.every_steps=1 \
checkpoint.every_steps=10000 \
optim.total_steps=50000000 \
model.klass="hf_t5"
```