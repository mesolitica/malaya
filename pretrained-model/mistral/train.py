from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    HfArgumentParser
)
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset, load_from_disk

import ray
from ray import train
from ray.train import Checkpoint
from ray.train.huggingface import TransformersTrainer
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
import os
import shutil
import wandb

wandb_api = wandb.Api()
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'run-ray')
WANDB_API_KEY = os.environ.get('WANDB_API_KEY', wandb_api.api_key)


class RayConnection:
    def __init__(self, address, **kwargs):
        ray.init(address=address, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        ray.shutdown()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "},
    )
    config_overrides: Optional[str] = field(
        default=None, metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index")}, )
    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"}, )
    use_fast_tokenizer: bool = field(
        default=True, metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}, )
    model_revision: str = field(
        default="main", metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."}, )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine.")}, )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."),
            "choices": [
                "auto",
                "bfloat16",
                "float16",
                "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "number of workers"},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
                self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."})
    train_file: Optional[str] = field(
        default=None, metadata={
            "help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None, metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}, )
    share_directory: str = field(
        default='/home/ubuntu/share',
        metadata={"help": "share storage directory"},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set.")}, )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set.")}, )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    checkpoint_steps: int = field(
        default=50,
        metadata={"help": "number of workers"},
    )


def train_func(config):

    get_device_str = str(get_device())
    os.environ['CUDA_VISIBLE_DEVICES'] = get_device_str.split(':')[-1]

    import torch
    from torch.utils.data import DataLoader
    import accelerate
    import transformers

    print(accelerate.__version__, transformers.__version__)

    torch.cuda.set_device(int(get_device_str.split(':')[-1]))
    local_rank = os.environ['LOCAL_RANK']
    node_rank = os.environ['NODE_RANK']
    print(f'node_rank: {node_rank}, local_rank: {local_rank}')

    model_args, data_args = config

    device = get_device_str.replace(':', '-')

    from streaming.base.format.mds.encodings import Encoding, _encodings
    from streaming import LocalDataset
    import streaming

    class UInt16(Encoding):
        def encode(self, obj) -> bytes:
            return obj.tobytes()

        def decode(self, data: bytes):
            return np.frombuffer(data, np.uint16)

    _encodings['uint16'] = UInt16

    class DatasetFixed(torch.utils.data.Dataset):
        def __init__(self, remote):

            streaming.base.util.clean_stale_shared_memory()
            self.dataset = LocalDataset(local=remote)

        def __getitem__(self, idx):
            data = self.dataset[idx]
            data['labels'] = data['input_ids'].copy()

            data.pop('token_type_ids', None)

            for k in data.keys():
                data[k] = data[k].astype(np.int64)
            return data

        def __len__(self):
            return len(self.dataset)

    directory = model_args.model_name_or_path.replace('/', '-')
    output_dir = os.path.join(data_args.share_directory, directory)

    train_dataset = DatasetFixed(remote=data_args.train_file)

    # https://github.com/mosaicml/streaming/issues/307#issuecomment-1729829065
    def inf_loop_dataloader(dataloader: torch.utils.data.DataLoader):
        while True:
            for batch in dataloader:
                yield batch

    dataloader = DataLoader(train_dataset, batch_size=2)
    dataset_iterator = iter(inf_loop_dataloader(dataloader))
    batch = next(iter(dataset_iterator))

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ['auto', None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        use_flash_attention_2=True,
        torch_dtype=torch_dtype,
    )
    model.gradient_checkpointing_enable()

    deepspeed = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "bf16": {
            "enabled": "auto"
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },

        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            }
        },

        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },

        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }

    training_args = TrainingArguments(
        output_dir,
        per_device_train_batch_size=24,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_strategy='steps',
        save_steps=data_args.checkpoint_steps,
        num_train_epochs=3,
        learning_rate=1e-4,
        weight_decay=1e-1,
        warmup_steps=2000,
        bf16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed,
        save_total_limit=5,
        log_level='info',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)

    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint

    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()
    except Exception as e:
        e = str(e)
        if checkpoint and 'checkpoint' in e.lower():
            os.system(f'mv {checkpoint} {checkpoint}-temp')


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    runtime_env = {
        'env_vars': {
            'WANDB_PROJECT': WANDB_PROJECT,
            'WANDB_API_KEY': WANDB_API_KEY,
            'TORCH_DISTRIBUTED_DEBUG': 'DETAIL',
            'NCCL_DEBUG': 'DEBUG'
        }
    }

    with RayConnection("ray://localhost:10001", runtime_env=runtime_env):
        scaling_config = ScalingConfig(
            num_workers=model_args.num_workers,
            use_gpu=True,
        )
        run_config = train.RunConfig(
            storage_path='/tmp/ray_results',
            failure_config=train.FailureConfig(
                max_failures=-1))
        ray_trainer = TorchTrainer(
            train_func,
            train_loop_config=(model_args, data_args),
            scaling_config=scaling_config,
            run_config=run_config

        )
        result = ray_trainer.fit()


if __name__ == "__main__":
    main()
