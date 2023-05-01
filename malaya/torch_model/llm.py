import torch
from malaya.function.streaming import Iteratorize, Stream
from malaya.text.prompt import template_alpaca as template
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
import logging

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


class LLM:
    def __init__(self, base_model, model, lora=True, **kwargs):
        if lora:
            validate_peft()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if self.device == 'cuda':
            device_map = 'auto'
            torch_dtype = torch.float16
        else:
            device_map = {'': self.device}
            torch_dtype = 'auto'

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model if lora else model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **kwargs,
        )

        if lora:
            self.model = PeftModel.from_pretrained(
                self.model,
                model,
                torch_dtype=torch.float16,
                **kwargs,
            )

        if 'gpt-j' in base_model:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        elif 'pythia' in base_model:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 1
            self.model.config.eos_token_id = self.tokenizer.eos_token_id = 0
        else:
            raise ValueError('only support `gpt-j` and `pythia` right now.')

        if self.device == 'cuda':
            self.model.half()

        _ = self.model.eval()

    def _get_input(self, query):
        prompt = template['prompt_no_input'].format(instruction=query)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        return input_ids

    def generate(
        self,
        query: str,
        temperature: float = 0.7,
        top_p: float = 0.75,
        top_k: int = 40,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        **kwargs,
    ):
        """
        Generate respond from user input.

        Parameters
        ----------
        query: str
            User input.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: str
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
        query: str,
        temperature: float = 0.7,
        top_p: float = 0.75,
        top_k: int = 40,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        **kwargs,
    ):
        """
        Generate respond from user input in streaming mode.

        Parameters
        ----------
        query: str
            User input.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: str
        """
        input_ids = self._get_input(query=query)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
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
