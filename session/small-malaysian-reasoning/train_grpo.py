import re
import click
import torch
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def length_reward_func(completions, **kwargs):
    """Reward function that gives higher scores to longer completions."""
    return [float(len(completion[0]["content"].split()) / 4096) for completion in completions]

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def correct_reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion[0]["content"]) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

@click.command()
@click.option('--model_name', default='huseinzol05/Llama-3.2-3B-Malaysian')
def main(model_name):
    dataset = load_dataset("huseinzol05/malay-reasoning", split="train")
    dataset = dataset.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': 'You are going to enter reasoning mode, first you try to think step-by-step in malay after that give the final answer.'},
            {'role': 'user', 'content': x['question']}
        ],
        'ground_truth': x['answer']
    })
    training_args = GRPOConfig(
        output_dir=f'output-{model_name}'.replace('/', '-'),
        run_name=f'GRPO-{model_name}'.replace('/', '-'),
        learning_rate=5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=8192,
        num_train_epochs=5,
        save_steps=100,
        max_grad_norm=0.1,
        report_to="wandb",
        log_on_each_node=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.8,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )
    print(training_args)
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
        bias="none",
        lora_dropout=0,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None
    )
    model = get_peft_model(model, peft_config)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            length_reward_func,
            format_reward_func,
            correct_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

if __name__ == "__main__":
    main()