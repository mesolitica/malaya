import torch
import logging
from malaya.function.streaming import Iteratorize, Stream
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList

logger = logging.getLogger(__name__)

peft_available = False
try:
    from peft import PeftModel

    peft_available = True
except Exception as e:
    logger.warning('`peft` is not available, not able to load any LLM models.')


def validate_peft():
    if not peft_available:
        raise ModuleNotFoundError(
            'peft not installed. Please install it by `pip install git+https://github.com/huggingface/peft` and try again.'
        )


template = {
    'description': 'Template used by Alpaca-LoRA.',
    'prompt_input': 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n',
    'prompt_no_input': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n',
    'response_split': '### Response:',
}


class LLM:
    def __init__(self, base_model, model):
        validate_peft()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if self.device == 'cuda':

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map='auto',
            )

            self.model = PeftModel.from_pretrained(
                self.model,
                model,
                torch_dtype=torch.float16,
            )

            self.model.half()

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map={'': self.device},
                low_cpu_mem_usage=True,
            )

            self.model = PeftModel.from_pretrained(
                self.model,
                model,
                device_map={'': self.device},
            )

        if 'gpt-j' in base_model:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        elif 'pythia' in base_model:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 1
            self.model.config.eos_token_id = self.tokenizer.eos_token_id = 0
        else:
            raise ValueError('only support `gpt-j` and `pythia` right now.')

        _ = self.model.eval()

    def _get_input(self, query):
        prompt = template['prompt_no_input'].format(instruction=query)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        return input_ids

    def generate(
        self,
        query,
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        """
        """
        input_ids = self._get_input(query=query)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                **kwargs,
            )
            s = generation_output.sequences[0]
            output = self.tokenizer.decode(s)
            return output.split(template['response_split'])[1].strip()

    def generate_stream(
        self,
        query,
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        """
        """
        input_ids = self._get_input(query=query)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
        )

        generate_params = {
            'input_ids': input_ids,
            'generation_config': generation_config,
            'return_dict_in_generate': True,
            'output_scores': True,
            'max_new_tokens': max_new_tokens,
        }

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                'stopping_criteria', StoppingCriteriaList()
            )
            kwargs['stopping_criteria'].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                self.model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                decoded_output = self.tokenizer.decode(output)
                if output[-1] in [self.tokenizer.eos_token_id]:
                    break

                yield output.split(decoded_output)[1].strip()
