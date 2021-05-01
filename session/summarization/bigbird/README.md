## how-to

1. BASE

```bash
python3 base.py \
--input_file=gs://mesolitica-tpu-general/t2t-summarization-2048/data/seq2* \
--output_dir=gs://mesolitica-tpu-general/bigbird-summarization-base \
--init_checkpoint=gs://mesolitica-tpu-general/pegasus-summarization-base/model.ckpt-600000 \
--do_train=True \
--train_batch_size=24 \
--num_train_steps=500000 \
--iterations_per_loop=100 \
--tpu_name=node-8 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=10000 \
--use_tpu=True
```

2. SMALL

```bash
python3 small.py \
--input_file=gs://mesolitica-tpu-general/t2t-summarization-2048/data/seq2* \
--output_dir=gs://mesolitica-tpu-general/bigbird-summarization-small \
--init_checkpoint=gs://mesolitica-tpu-general/pegasus-summarization-small/model.ckpt-740000 \
--do_train=True \
--train_batch_size=64 \
--num_train_steps=500000 \
--iterations_per_loop=100 \
--tpu_name=node-9 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=10000 \
--use_tpu=True
```

## download

1. **SMALL**, last update 19th February 2021, [abstractive-summarization-small-bigbird.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/abstractive-summarization-small-bigbird.tar.gz)

2. **BASE**, last update 19th February 2021, [abstractive-summarization-base-bigbird.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/abstractive-summarization-base-bigbird.tar.gz)