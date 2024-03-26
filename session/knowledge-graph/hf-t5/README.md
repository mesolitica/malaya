> The original script for fine-tuning the T5 model is available at https://github.com/huggingface/transformers/blob/v4.21.2/examples/pytorch/translation/run_translation.py

Knowledge graphs are structured representations of knowledge that capture the relationships between various entities and concepts. Using them, we are able to organize and store information in a form that can be easily understood and reasoned over by both humans and machines.

T5 (Text-to-Text Transfer Transformer) is a powerful encoder-decoder model introduced by Google AI. Learn more about it [here](https://huggingface.co/docs/transformers/en/model_doc/t5)

## Steps for Execution

1. Improve the existing dataset
2. Run `prepare-wikipedia.ipynb` and `prepare-astroawani.ipynb` to generate the processed kg triplets file
3. Run `prepare-data.ipynb`
4. Run `run_t5_v2.py` to start training and generate the checkpoints
5. Run `export-small.ipynb` as we're done fine-tuning the model to a specific task

Below are the command line environment variables and parameters that you'll be able to copy and paste to use depending on what model you want to train for:

### SMALL model

```bash
CUDA_VISIBLE_DEVICES='0' \
WANDB_DISABLED=true \
python3 run_t5_v2.py \
--model_name_or_path mesolitica/nanot5-small-malaysian-cased \
--num_train_epochs 5 \
--logging_steps 100 \
--eval_steps 1000000000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--bf16 \
--source_lang src \
--target_lang tgt \
--train_file shuf-train.json \
--validation_file test-4k.json \
--output_dir nanot5-small-malaysian-cased \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--max_source_length 1024 \
--max_target_length 1024 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

### BASE model

```bash
CUDA_VISIBLE_DEVICES='1' \
WANDB_DISABLED=true \
python3 run_t5_v2.py \
--model_name_or_path mesolitica/nanot5-base-malaysian-cased \
--num_train_epochs 5 \
--logging_steps 100 \
--eval_steps 1000000000 \
--save_steps 10000 \
--evaluation_strategy steps \
--save_total_limit 3 \
--do_train \
--do_eval \
--bf16 \
--source_lang src \
--target_lang tgt \
--train_file shuf-train.json \
--validation_file test-4k.json \
--output_dir nanot5-base-malaysian-cased \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--max_source_length 1024 \
--max_target_length 1024 \
--learning_rate 2e-4 \
--gradient_checkpointing true
```

Here are the explanations for each parameter:

`CUDA_VISIBLE_DEVICES='0'`: This specifies which GPU device(s) should be used for training. In this case, it's set to '0', which means the first GPU device will be used.
`WANDB_DISABLED=true`: This disables the weight and biases (W&B) logging.
`--model_name_or_path mesolitica/nanot5-small-malaysian-cased`: This specifies the pre-trained T5 model to be used
`--num_train_epochs 5`: This sets the number of training epochs to 5.
`--logging_steps 100`: This sets the logging frequency to log every 100 steps during training.
`--eval_steps 1000000000`: This sets the evaluation frequency to an extremely large value, effectively disabling periodic evaluation during training.
`--save_steps 10000`: This saves the model checkpoint every 10,000 steps during training.
`--evaluation_strategy steps`: This specifies that the evaluation strategy is based on steps (number of updates) rather than epochs.
`--save_total_limit 3`: This limits the total number of checkpoints to be saved to 3.
`--do_train` and `--do_eval`: These flags enable training and evaluation modes, respectively.
`--bf16`: This enables Brain Float 16 (BF16) precision for faster training on compatible hardware.
`--source_lang src` and `--target_lang tgt`: These specify the source and target languages for the translation task.
`--train_file shuf-train.json` and `--validation_file test-4k.json`: These specify the paths to the training and validation data files in JSON format.
`--output_dir nanot5-small-malaysian-cased`: This sets the output directory for saving the fine-tuned model and logs.
`--per_device_train_batch_size=8` and `--per_device_eval_batch_size=4`: These set the batch sizes for training and evaluation, respectively.
`--predict_with_generate`: This enables the use of the generate method for prediction during evaluation.
`--max_source_length 1024` and `--max_target_length 1024`: These set the maximum lengths for the source and target sequences, respectively.
`--learning_rate 2e-4`: This sets the learning rate for the optimizer.
`--gradient_checkpointing true`: This enables gradient checkpointing, which is a memory-saving technique for large models.
