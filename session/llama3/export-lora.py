import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from transformers.trainer_utils import get_last_checkpoint

latest = get_last_checkpoint("/workspace/unsloth-Meta-Llama-3.1-8B-Instruct-lora-128-embedding-8k-multipack")
latest

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel

model = AutoPeftModelForCausalLM.from_pretrained(latest)
model.push_to_hub('mesolitica/lora-embedding-128-llama3.1-8b-multipack-v2', private = True)