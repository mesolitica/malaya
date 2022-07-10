import tensorflow as tf
from herpetologist import check_type
from typing import List


class Generator:
    def __init__(self, model, tokenizer, initial_text='', **kwargs):
        self._model = model
        self._tokenizer = tokenizer
        self._initial_text = initial_text

    @check_type
    def generate(self, strings: List[str], **kwargs):
        """
        Generate texts from the input.

        Parameters
        ----------
        strings : List[str]
        **kwargs: vector arguments pass to huggingface `generate` method.

        Returns
        -------
        result: List[str]
        """
        input_ids = [{'input_ids': self._tokenizer.encode(f'{self._initial_text}{s}', return_tensors='tf')[
            0]} for s in strings]
        padded = self._tokenizer.pad(input_ids, padding='longest')
        outputs = self._model.generate(**padded, **kwargs)
        results = []
        for o in outputs:
            results.append(self._tokenizer.decode(o, skip_special_tokens=True))
        return results
