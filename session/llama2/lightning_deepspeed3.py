import os
os.environ['REDIS_PORT'] = '6379'
os.system('sudo apt install ninja-build -y')

import argparse
import torch
import time
from torch.utils.data import DataLoader
from itertools import chain
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    default_data_collator,
    SchedulerType
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DeepSpeedStrategy
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam


def prepare_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    block_size = args.block_size
    text_column_name = 'text'

    raw_datasets = load_dataset(
        'json',
        data_files=args.train_file,
        split='train'
    )

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    column_names = raw_datasets.column_names
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return DataLoader(
        lm_datasets,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )


class Module(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
        if args.compile:
            self.model = torch.compile(self.model)
        self.args = args

    def configure_optimizers(self):
        self.optimizer = DeepSpeedCPUAdam(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay)
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
        )
        scheduler = {"scheduler": self.lr_scheduler, "interval": "step", "frequency": 1}
        return (
            [self.optimizer],
            [scheduler],
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "Losses/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--train_file", type=str, default=None,
                        help="a json file containing the training data.")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="a json file containing the validation data.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default='bf16',
        help="precision",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="gradient clipping",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--compile",
        type=bool,
        default=False,
        help='pytorch 2.0 compile',
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=1024,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
            "execute code present on the Hub on your local machine."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=100,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."),
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    train_dataloader = prepare_dataset(args)
    model = Module(args)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor='step',
        mode='max',
        dirpath=args.output_dir,
        every_n_train_steps=args.checkpointing_steps,
        filename='model-{epoch:02d}-{step}',
    )
    num_gpus = torch.cuda.device_count()

    trainer = pl.Trainer(
        max_steps=args.max_train_steps,
        gradient_clip_val=args.grad_clip,
        accelerator='gpu',
        devices=num_gpus,
        limit_val_batches=100,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        ),
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=None,
    )


if __name__ == "__main__":
    main()
