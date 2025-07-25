from trl import GRPOConfig, GRPOTrainer, TrlParser
import re
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sacrebleu.metrics import CHRF
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass, field
from typing import Optional

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
    rank: int = field(
        default=256,
        metadata={
            "help": "lora rank"
        },
    )
    alpha: int = field(
        default=None,
        metadata={
            "help": "lora rank alpha"
        },
    )
    use_rslora: bool = field(
        default=False,
        metadata={
            "help": "use_rslora"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "dataset"
            )
        },
    )

def contains_chinese(completions, **kwargs):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return [int(not bool(chinese_pattern.search(completion[0]["content"]))) for completion in completions]

def dialect_func(completions, ground_truth, task, **kwargs):
    """
    answers = re.findall(r"\$boxed\{(.*?)\}\$", text)
    score = chrf.corpus_score([answers[0]], [[answer]]).score
    """
    chrf = CHRF()
    matches = [re.search(r"\$boxed\{(.*?)\}\$", completion[0]["content"]) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    scores = []
    for c, gt, t in zip(contents, ground_truth, task):
        if t == "dialect":
            score = chrf.corpus_score([c], [[gt]]).score
            print('dialect', c, gt, score)
            scores.append(score / 100.0)
        else:
            scores.append(None)
    return scores

def translation_func(completions, ground_truth, task, **kwargs):
    """
    answers = re.findall(r"\$boxed\{(.*?)\}\$", text)
    score = chrf.corpus_score([answers[0]], [[answer]]).score
    """
    chrf = CHRF()
    matches = [re.search(r"\$boxed\{(.*?)\}\$", completion[0]["content"]) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    scores = []
    for c, gt, t in zip(contents, ground_truth, task):
        if t == "translation":
            score = chrf.corpus_score([c], [[gt]]).score
            print('translation', c, gt, score)
            scores.append(score / 100.0)
        else:
            scores.append(None)
    return scores

def qa_func(completions, ground_truth, task, **kwargs):
    """
    answers = re.findall(r"\$boxed\{(.*?)\}\$", text)
    score = chrf.corpus_score([answers[0]], [[answer]]).score
    """
    chrf = CHRF()
    matches = [re.search(r"\$boxed\{(.*?)\}\$", completion[0]["content"]) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    scores = []
    for c, gt, t in zip(contents, ground_truth, task):
        if t == "qa":
            score = int(c == gt)
            print('qa', c, gt, score)
            scores.append(score)
        else:
            scores.append(None)
    return scores

format = lambda x: {
    'prompt': [
        {
            "role": "system",
            "content": 'You are going to enter reasoning mode. First, you try to think step-by-step in Malay. After that, put your final answer within $\\boxed{}$.',
        },
        {
            'role': 'user',
            'content': x['question']
        }
    ],
    'ground_truth': x['answer'],
    'task': x['task']
}

def main(model_args, data_args, training_args):
    train_dataset = load_dataset(data_args.dataset, split="train")
    train_dataset = train_dataset.map(format)
    
    """
    training_args = GRPOConfig(
        output_dir=f'output-{model_name}'.replace('/', '-'),
        run_name=f'GRPO-{model_name}'.replace('/', '-'),
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_steps = 50,
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=8192,
        num_train_epochs=5,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        use_vllm=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        deepspeed='ds_config_zero3.json',
        use_liger_loss=True,
    )
    """
    
    print(training_args)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="sdpa",
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    if model_args.rank > 0:
        if model_args.alpha is None:
            alpha = model_args.rank * 2

        peft_config = LoraConfig(
            lora_alpha=alpha,
            lora_dropout=0.0,
            r=model_args.rank,
            use_rslora=model_args.use_rslora,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            contains_chinese,
            dialect_func,
            translation_func,
            qa_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
    )
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

if __name__ == "__main__":
    parser = TrlParser((ModelArguments, DataTrainingArguments, GRPOConfig,))
    model_args, data_args, training_args = parser.parse_args_and_config()
    main(model_args, data_args, training_args)