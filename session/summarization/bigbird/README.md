## how-to

1. BASE

```bash
python3 base.py \
--input_file=gs://mesolitica-tpu-general/t2t-summarization/data/seq2* \
--output_dir=gs://mesolitica-tpu-general/bigbird-summarization-base \
--init_checkpoint=gs://mesolitica-tpu-general/pegasus-base-multitasks/model.ckpt-1000000 \
--do_train=True \
--train_batch_size=32 \
--num_train_steps=50000 \
--iterations_per_loop=100 \
--tpu_name=node-2 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=25000 \
--use_tpu=True
```

1. SMALL

```bash
python3 small.py \
--input_file=gs://mesolitica-tpu-general/t2t-summarization/data/seq2* \
--output_dir=gs://mesolitica-tpu-general/bigbird-summarization-small \
--init_checkpoint=gs://mesolitica-tpu-general/pegasus-small-multitasks/model.ckpt-1000000 \
--do_train=True \
--train_batch_size=64 \
--num_train_steps=50000 \
--iterations_per_loop=100 \
--tpu_name=node-1 \
--tpu_zone=europe-west4-a \
--save_checkpoints_steps=25000 \
--use_tpu=True
```