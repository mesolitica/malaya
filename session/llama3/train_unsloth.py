import streaming
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from streaming import LocalDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from transformers import (
    TrainingArguments, 
    HfArgumentParser, 
    Trainer, 
    default_data_collator
)
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers.trainer_utils import get_last_checkpoint
import os
import torch

@dataclass
class UnslothTrainingArguments(TrainingArguments):
    embedding_learning_rate : Optional[float] = field(
        default = None,
        metadata = {"help" : "Different learning rates for embeddings and lm_head."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    context_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Context length"
            )
        },
    )

    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "dataset"
            )
        },
    )
    

def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

class UnslothTrainer(Trainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        pass
        return self.optimizer
    pass
pass

class UInt32(Encoding):
    def encode(self, obj) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes):
        return np.frombuffer(data, np.uint32)

_encodings['uint32'] = UInt32

class DatasetFixed(torch.utils.data.Dataset):
    def __init__(self, local):
        self.dataset = LocalDataset(local=local)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data['labels'] = data["input_ids"].copy()
        data.pop('token_type_ids', None)
        for k in data.keys():
            data[k] = data[k].astype(np.int64)
        return data

    def __len__(self):
        return len(self.dataset)
        
def main():
    parser = HfArgumentParser((ModelArguments, UnslothTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_args.model_name_or_path,
        max_seq_length = model_args.context_length,
        dtype = None,
        load_in_4bit = True,
    )

    global_rank = int(os.environ['RANK'])

    model = FastLanguageModel.get_peft_model(
        model,
        r = 128,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none", 
        use_gradient_checkpointing = 'unsloth',
        random_state = 3407,
        max_seq_length = model_args.context_length,
        use_rslora = True,
        loftq_config = None,
        temporary_location = f'temp_{global_rank}'
    )

    dataset = DatasetFixed(model_args.dataset)

    trainer = UnslothTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    checkpoint = get_last_checkpoint(training_args.output_dir)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

if __name__ == "__main__":
    main()