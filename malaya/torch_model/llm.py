"""
Mostly borrowed from https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/model.py
"""

import torch
from threading import Thread
from malaya.text.prompt import template_alpaca, template_malaya
from malaya_boilerplate.converter import ctranslate2_generator
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import TextIteratorStreamer
except BaseException:
    logger.warning(
        '`transformers.TextIteratorStreamer` is not available, `malaya.torch_model.LLM.generate_stream` is not able to use.')
    TextIteratorStreamer = None


class LLM:
    def __init__(self, model, use_ctranslate2=False, **kwargs):

        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.use_ctranslate2 = use_ctranslate2

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
            model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            **kwargs,
        )

        _ = self.model.eval()

    def _get_input(self, query):
        prompt = template['prompt_no_input'].format(instruction=query)
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        return input_ids

    def generate(
        self,
        query: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        do_sample=True,
        template=template_malaya,
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
                do_sample=do_sample,
                **kwargs,
            )
            s = generation_output.sequences[0]
            output = self.tokenizer.decode(s)
            return output.split(template['response_split'])[1].strip()

    def generate_stream(
        self,
        query: str,
        temperature: float = 0.9,
        top_p: float = 0.75,
        top_k: int = 40,
        num_beams: int = 4,
        max_new_tokens: int = 128,
        do_sample: bool = True,
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
        if TextIteratorStreamer is None:
            raise ModuleNotFoundError(
                '`transformers.TextIteratorStreamer` is not available, please install latest transformers library.'
            )
        input_ids = self._get_input(query=query)
        streamer = TextIteratorStreamer(
            tokenizer,
            timeout=10.,
            skip_prompt=True,
            skip_special_tokens=True,
        )
