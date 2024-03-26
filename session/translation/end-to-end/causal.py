import os

os.environ['LD_LIBRARY_PATH'] = '/home/husein/.local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib'
os.environ["WANDB_DISABLED"] = "true"

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import fire


def main(
    base_model='stabilityai/stablelm-base-alpha-7b',
    cutoff_len=1536,
    batch_size=64,
    micro_batch_size=4,
    learning_rate: float = 3e-4,
    save_steps=100,
    num_epochs=3,
    val_set_size: int = 0,
    output_dir: str = "./lora-gptj",
    data_files='combined.jsonl',
    group_by_length=False,
    use_wandb=False,
    wandb_run_name='',
    resume_from_checkpoint=None,
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        offload_folder="offload",
        offload_state_dict=True,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 0
    tokenizer.padding_side = "right"

    template = '{instruction}\n### Response:\n'

    def tokenize(prompt, add_eos_token=True):
        prefix = prompt['prefix']
        src = prompt['src']
        instruction = f'{prefix}{src}'
        prompt = template.format(instruction=instruction) + prompt['tgt']
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point['input']
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_files)
    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].map(
            generate_and_tokenize_prompt,
            load_from_cache_file=True,
            cache_file_name='./translation')
        val_data = None

    print(train_data, val_data)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=2,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=1 if val_set_size > 0 else None,
            save_steps=save_steps,
            gradient_checkpointing=True,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
