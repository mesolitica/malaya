import os

os.environ['HF_HOME'] = "/workspace/cache"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from transformers.trainer_utils import get_last_checkpoint

latest = get_last_checkpoint("/workspace/Qwen-Qwen2.5-14B-Instruct-lora-128-embedding-8k-multipack")
latest

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-14B-Instruct', torch_dtype = torch.bfloat16)
m = PeftModel.from_pretrained(model, latest, torch_dtype = torch.bfloat16)
m.push_to_hub('mesolitica/Qwen-Qwen2.5-14B-Instruct-lora-128-embedding-8k-multipack', private = True)