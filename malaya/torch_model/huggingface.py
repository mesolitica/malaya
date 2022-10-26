from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy
from herpetologist import check_type
from typing import List


class Base:
    def cuda(self, **kwargs):
        return self.model.cuda(**kwargs)


class Generator(Base):
    def __init__(self, model, initial_text, use_fast_tokenizer=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast_tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self._initial_text = initial_text

    @check_type
    def generate(self, strings: List[str], **kwargs):
        """
        Generate texts from the input.

        Parameters
        ----------
        strings : List[str]
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        cuda = next(self.model.parameters()).is_cuda
        input_ids = [{'input_ids': self.tokenizer.encode(f'{self._initial_text}{s}', return_tensors='pt')[
            0]} for s in strings]
        padded = self.tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)
        outputs = self.model.generate(**padded, **kwargs)
        results = []
        for o in outputs:
            results.append(self.tokenizer.decode(o, skip_special_tokens=True))
        return results


class Prefix(Base):
    def __init__(self, model, use_fast_tokenizer=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast_tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(model)

    @check_type
    def generate(self, string, **kwargs):
        """
        Generate texts from the input.

        Parameters
        ----------
        string : str
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        cuda = next(self.model.parameters()).is_cuda
        padded = {'input_ids': tokenizer.encode(string, return_tensors='pt')}
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)
        outputs = model.generate(**padded, **kwargs)
        results = []
        for o in outputs:
            results.append(tokenizer.decode(o, skip_special_tokens=True))
        return results
