import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from transformers.trainer_utils import get_last_checkpoint

latest = get_last_checkpoint("/root/lora-embedding-256-llama3.2-3b-multipack")
latest

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel

model = AutoPeftModelForCausalLM.from_pretrained(latest)
model.push_to_hub('mesolitica/lora-embedding-256-llama3.2-3b-multipack', private = True)