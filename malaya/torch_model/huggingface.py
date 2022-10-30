from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from malaya.text.rouge import postprocess_summary, find_kata_encik
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


class Paraphrase(Generator):
    def __init__(self, model, initial_text, use_fast_tokenizer=False):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
            use_fast_tokenizer=use_fast_tokenizer,
        )

    @check_type
    def generate(
        self,
        strings: List[str],
        postprocess: bool = True,
        **kwargs,
    ):
        """
        Generate texts from the input.

        Parameters
        ----------
        strings : List[str]
        postprocess: bool, optional (default=False)
            If True, will removed biased generated `kata Encik`.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        results = super().generate(strings, **kwargs)
        if postprocess:
            for no in range(len(results)):
                s = find_kata_encik(strings[no])
                results[no] = s
        return results


class Summarization(Generator):
    def __init__(self, model, initial_text, use_fast_tokenizer=False):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
            use_fast_tokenizer=use_fast_tokenizer,
        )

    @check_type
    def generate(
        self,
        strings: List[str],
        postprocess: bool = True,
        n: int = 2,
        threshold: float = 0.1,
        reject_similarity: float = 0.85,
        **kwargs,
    ):
        """
        Generate texts from the input.

        Parameters
        ----------
        strings : List[str]
        postprocess: bool, optional (default=False)
            If True, will filter sentence generated using ROUGE score and removed biased generated international news publisher.
        n: int, optional (default=2)
            N size of rouge to filter
        threshold: float, optional (default=0.1)
            minimum threshold for N rouge score to select a sentence.
        reject_similarity: float, optional (default=0.85)
            reject similar sentences while maintain position.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        results = super().generate(strings, **kwargs)
        if postprocess:
            for no in range(len(results)):
                s = postprocess_summary(
                    strings[no],
                    results[no],
                    n=n,
                    threshold=threshold,
                    reject_similarity=reject_similarity,
                )
                results[no] = s
        return results
