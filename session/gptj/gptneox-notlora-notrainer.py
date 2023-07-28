import os
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM, get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
import fire
import math
from tqdm import tqdm

logger = get_logger(__name__)


def main(
    base_model='stabilityai/stablelm-base-alpha-7b',
    cutoff_len=1536,
    batch_size=4,
    gradient_accumulation_steps=32,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    save_steps=100,
    num_train_epochs=3,
    output_dir: str = "./lora-gptj",
    data_files='combined.jsonl',
    resume_from_checkpoint=None,
    warmup_steps=100,
    logging_steps=2,
    with_tracking=True,
):
    accelerator_log_kwargs = {}
    accelerator_log_kwargs["log_with"] = 'tensorboard'
    accelerator_log_kwargs["logging_dir"] = output_dir
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    model = GPTNeoXForCausalLM.from_pretrained(
        base_model,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 1
    tokenizer.eos_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
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
    with accelerator.main_process_first():
        train_data = data["train"].map(
            generate_and_tokenize_prompt,
            load_from_cache_file=True,
            remove_columns=['input'],
            cache_file_name='./' + data_files + '_gptneox-notrainer-cache')
        val_data = None

    print(train_data, val_data)
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    train_dataloader = DataLoader(
        train_data, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = None

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(
                    nd in n for nd in no_decay)], "weight_decay": weight_decay, }, {
            "params": [
                p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], "weight_decay": 0.0, }, ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    overrode_max_train_steps = True
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    checkpointing_steps = save_steps

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if resume_from_checkpoint:
        if resume_from_checkpoint is not None or resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        if with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            if resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if step % logging_steps == 0:
                    print(step, loss.detach().float())
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir_ = f"step_{completed_steps }"
                    if output_dir is not None:
                        output_dir_ = os.path.join(output_dir, output_dir_)
                    accelerator.save_state(output_dir_)

            if completed_steps >= max_train_steps:
                break


if __name__ == "__main__":
    fire.Fire(main)
