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
CUDA_VISIBLE_DEVICES=0 python3 train_lightning.py \
optim.name=adamwscale optim.batch_size=144 \
optim.lr_scheduler=cosine optim.grad_acc=6 \
model.name="/home/husein/dev/malaya/pretrained-model/nanoT5/out-base-1.1" \
model.random_init=false \
data.filename.train="/home/husein/dev/malaya/pretrained-model/nanoT5/combine.jsonl" \
data.filename.test="/home/husein/dev/malaya/pretrained-model/nanoT5/combine.jsonl" \
checkpoint.every_steps=10000 optim.total_steps=500000 \
model.klass="hf_t5" model_directory="base-v3" \
model.compile=true optim.base_lr=2e-3 optim.warmup_steps=30000 \
model.checkpoint_path="/home/husein/dev/malaya/pretrained-model/nanoT5/logs/base-v3/model-epoch\=00-step\=50000.ckpt"
```

- SMALL,

```bash
CUDA_VISIBLE_DEVICES=1 python3 train_lightning.py \
optim.name=adamwscale \
optim.batch_size=128 \
optim.lr_scheduler=cosine \
optim.grad_acc=2 \
model.name="/home/husein/dev/malaya/pretrained-model/nanoT5/out-small-1.1" \
model.random_init=false \
data.filename.train="/home/husein/dev/malaya/pretrained-model/nanoT5/combine-others.jsonl" \
data.filename.test="/home/husein/dev/malaya/pretrained-model/nanoT5/combine-others.jsonl" \
checkpoint.every_steps=5000 \
optim.total_steps=500000 \
model.klass="hf_t5" \
model_directory="small-v2" \
model.compile=true \
model.checkpoint_path="/home/husein/dev/malaya/pretrained-model/nanoT5/logs/small-v2/model-epoch\=00-step\=170000.ckpt"
```

- TINY,

```bash
python3 train_lightning.py \
optim.name=adamwscale \
optim.batch_size=128 \
optim.lr_scheduler=cosine \
optim.grad_acc=2 \
model.name="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/out-tiny-1.1" \
model.random_init=false \
data.filename.train="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/combine-others.jsonl" \
data.filename.test="/home/ubuntu/server2/dev/malaya/pretrained-model/nanoT5/combine-others.jsonl" \
checkpoint.every_steps=10000 \
optim.total_steps=500000 \
model.klass="hf_t5" \
model_directory="tiny" \
model.compile=true
```

- SUPER TINY,

```bash
CUDA_VISIBLE_DEVICES=1 \
python3 train_lightning.py \
optim.name=adamwscale \
optim.batch_size=192 \
optim.lr_scheduler=cosine \
optim.grad_acc=1 \
model.name="/home/husein/dev/malaya/pretrained-model/nanoT5/out-super-tiny-1.1" \
model.random_init=false \
data.filename.train="/home/husein/dev/malaya/pretrained-model/nanoT5/combine.jsonl" \
data.filename.test="/home/husein/dev/malaya/pretrained-model/nanoT5/combine.jsonl" \
checkpoint.every_steps=10000 \
optim.total_steps=500000 \
model.klass="hf_t5" \
model_directory="super-tiny" \
model.compile=true
```