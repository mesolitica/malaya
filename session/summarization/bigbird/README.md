## how-to

1. BASE

```bash
python3 base.py \
--input_file=gs://mesolitica-tpu-general/t2t-summarization-v2/data/seq2* \
--output_dir=gs://mesolitica-tpu-general/bigbird-summarization-base \
--init_checkpoint=gs://mesolitica-tpu-general/pegasus-base-multitasks/model.ckpt-1000000 \
--do_train=True \
--train_batch_size=32 \
--num_train_steps=50000 \
--iterations_per_loop=100 \
--tpu_name=node-1 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=10000 \
--use_tpu=True
```

2. SMALL

```bash
python3 small.py \
--input_file=gs://mesolitica-tpu-general/t2t-summarization-v2/data/seq2* \
--output_dir=gs://mesolitica-tpu-general/bigbird-summarization-small \
--init_checkpoint=gs://mesolitica-tpu-general/pegasus-small-multitasks/model.ckpt-1000000 \
--do_train=True \
--train_batch_size=64 \
--num_train_steps=50000 \
--iterations_per_loop=100 \
--tpu_name=node-2 --tpu_zone=europe-west4-a \
--save_checkpoints_steps=10000 \
--use_tpu=True
```

## download

1. **SMALL**, last update 19th February 2021, [abstractive-summarization-small-bigbird.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/abstractive-summarization-small-bigbird.tar.gz)

2. **BASE**, last update 19th February 2021, [abstractive-summarization-base-bigbird.tar.gz](https://f000.backblazeb2.com/file/malaya-model/pretrained/abstractive-summarization-base-bigbird.tar.gz)