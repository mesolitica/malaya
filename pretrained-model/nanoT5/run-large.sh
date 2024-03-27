rm -rf /dev/shm/*
WANDB_PROJECT=nanoT5-large \
~/.local/bin/torchrun --nproc_per_node 4 \
-m train \
model.name="/home/ubuntu/malaya/pretrained-model/nanoT5/out-large-1.1" \
model.random_init=false \
data.filename.train="/home/ubuntu/nanot5-512" \
data.filename.test="/home/ubuntu/nanot5-512" \
data.input_length=512 \
checkpoint.every_steps=400 \
optim.total_steps=65536 \
optim.name=adamwscale \
optim.batch_size=128 \
optim.lr_scheduler=cosine \
optim.grad_acc=8 \
optim.grad_clip=1.0 \
optim.warmup_steps=2500 \
optim.base_lr=2e-2 \
model.klass="hf_t5" \
model_directory="large"