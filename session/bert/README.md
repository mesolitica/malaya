## How-to

1. Execute pretraining,
```bash
python3 run_pretraining.py --input_file=texts_output.tfrecord --output_dir=pretraining_output --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json --train_batch_size=8 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=50000 --num_warmup_steps=10 --learning_rate=2e-5 --save_checkpoints_steps=10000
```
