## how-to

1. BASE

```bash
python3 base.py \
--input_file=gs://mesolitica-tpu-general/pegasus-summarization-data/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/pegasus-summarization-base \
--init_checkpoint=gs://mesolitica-tpu-general/pegasus-base-v3/model.ckpt-1500000 \
--do_train=True \
--train_batch_size=128 \
--num_train_steps=100000 \
--iterations_per_loop=100 \
--tpu_name=node-3 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=10000 \
--use_tpu=True
```

2. SMALL

```bash
python3 small.py \
--input_file=gs://mesolitica-tpu-general/pegasus-summarization-data/*.tfrecord \
--output_dir=gs://mesolitica-tpu-general/pegasus-summarization-small \
--init_checkpoint=gs://mesolitica-tpu-general/pegasus-small-v3/model.ckpt-1875000 \
--do_train=True \
--train_batch_size=256 \
--num_train_steps=100000 \
--iterations_per_loop=100 \
--tpu_name=node-8 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=10000 \
--use_tpu=True
```

## Download

1. SMALL, last update 27th July 2021, [pegasus-summarization-small-2021-07-27.tar.gz](https://f000.backblazeb2.com/file/malaya-model/finetuned/pegasus-summarization-small-2021-07-27.tar.gz)

2. BASE, last update 25th April 2021, [pegasus-summarization-base-2021-07-27.tar.gz](https://f000.backblazeb2.com/file/malaya-model/finetuned/pegasus-summarization-base-2021-07-27.tar.gz)