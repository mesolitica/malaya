from transformers import T5Tokenizer, T5ForConditionalGeneration
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy
from herpetologist import check_type
from typing import List


class Generator(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def load_tokenizer(self, model_name=None, use_fast_tokenizer=False):
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name or self.config._name_or_path, use_fast=use_fast_tokenizer)

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
        cuda = next(self.parameters()).is_cuda
        input_ids = [{'input_ids': self.tokenizer.encode(f'{self._initial_text}{s}', return_tensors='pt')[
            0]} for s in strings]
        padded = self.tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)
        outputs = super().generate(**padded, **kwargs)
        results = []
        for o in outputs:
            results.append(self.tokenizer.decode(o, skip_special_tokens=True))
        return results
