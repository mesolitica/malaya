from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    RobertaTokenizer,
    ElectraTokenizer,
    BertTokenizer,
    T5Tokenizer,
    AlbertTokenizer,
    XLNetTokenizer,
)
from malaya.text.bpe import (
    merge_sentencepiece_tokens_tagging,
    merge_sentencepiece_tokens,
    merge_wordpiece_tokens,
    merge_bpe_tokens,
)
from malaya.text.function import (
    summarization_textcleaning,
    upperfirst,
    remove_repeat_fullstop,
    remove_newlines,
)
from malaya.text.rouge import postprocess_summary, find_kata_encik
from malaya.torch_model.t5 import T5ForSequenceClassification, T5Tagging
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy
from collections import defaultdict
from herpetologist import check_type
from typing import List
import numpy as np
import torch
import re
import logging

logger = logging.getLogger(__name__)


class Base:
    def cuda(self, **kwargs):
        return self.model.cuda(**kwargs)


class Generator(Base):
    def __init__(self, model, initial_text, base_model=AutoModelForSeq2SeqLM, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = base_model.from_pretrained(model, **kwargs)
        self._initial_text = initial_text

    @check_type
    def generate(self, strings: List[str], return_generate=False, **kwargs):
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

        logger.debug(f'generate, strings: {strings}')

        cuda = next(self.model.parameters()).is_cuda
        input_ids = [{'input_ids': self.tokenizer.encode(f'{self._initial_text}{s}', return_tensors='pt')[
            0]} for s in strings]
        padded = self.tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)
        outputs = self.model.generate(**padded, **kwargs)
        if return_generate:
            return outputs
        else:
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class Prefix(Base):
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)

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
        outputs = self.model.generate(**padded, **kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class Paraphrase(Generator):
    def __init__(self, model, initial_text, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
            **kwargs,
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
    def __init__(self, model, initial_text, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
            **kwargs,
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


class Similarity(Base):
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = T5ForSequenceClassification.from_pretrained(model, **kwargs)

    def forward(self, strings_left: List[str], strings_right: List[str]):
        if len(strings_left) != len(strings_right):
            raise ValueError('len(strings_left) != len(strings_right)')

        cuda = next(self.model.parameters()).is_cuda

        strings = []
        for i in range(len(strings_left)):
            s1 = strings_left[i]
            s2 = strings_right[i]
            s = f'ayat1: {s1} ayat2: {s2}'
            strings.append(s)

        input_ids = [{'input_ids': self.tokenizer.encode(s, return_tensors='pt')[0]} for s in strings]
        padded = self.tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)

        outputs = self.model(**padded, return_dict=True)
        return outputs

    @check_type
    def predict_proba(self, strings_left: List[str], strings_right: List[str]):
        """
        calculate similarity for two different batch of texts.

        Parameters
        ----------
        strings_left : List[str]
        strings_right : List[str]

        Returns
        -------
        list: List[float]
        """

        outputs = self.forward(strings_left=strings_left, strings_right=strings_right)
        entail_contradiction_logits = outputs.logits
        probs = entail_contradiction_logits.softmax(dim=1)[:, 1]
        return to_numpy(probs)


class ZeroShotClassification(Similarity):
    def __init__(self, model, **kwargs):
        Similarity.__init__(
            self,
            model=model,
            **kwargs
        )

    def predict_proba(
        self,
        strings: List[str],
        labels: List[str],
        prefix: str = 'ayat ini berkaitan tentang'
    ):
        """
        classify list of strings and return probability.

        Parameters
        ----------
        strings: List[str]
        labels: List[str]
        prefix: str
            prefix of labels to zero shot. Playing around with prefix can get better results.

        Returns
        -------
        list: List[Dict[str, float]]
        """
        strings_left, strings_right, mapping = [], [], defaultdict(list)
        index = 0
        for no, string in enumerate(strings):
            for label in labels:
                strings_left.append(string)
                text_label = f'{prefix} {label}'
                text_label = re.sub(r'[ ]+', ' ', text_label).strip()
                strings_right.append(text_label)
                mapping[no].append(index)
                index += 1

        outputs = super().forward(strings_left=strings_left, strings_right=strings_right)
        entail_contradiction_logits = outputs.logits[:, [0, 1]]
        probs = to_numpy(entail_contradiction_logits.softmax(dim=1)[:, 1])
        results = []
        for k, v in mapping.items():
            result = {}
            for no, index in enumerate(v):
                result[labels[no]] = probs[index]
            results.append(result)
        return results


class ZeroShotNER(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            **kwargs,
        )
        self._q = 'Ekstrak <entity> dari teks: '
        self._placeholder = '<entity>'

    def predict(
        self,
        string: str,
        tags: List[str],
        minimum_length: int = 2,
        **kwargs,
    ):
        """
        classify entities in a string.

        Parameters
        ----------
        strings: str
            We assumed the string input been properly tokenized.
        tags: List[str]
        minimum_length: int, optional (default=2)
            minimum length of string for an entity.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        list: Dict[str, List[str]]
        """

        strings = []
        for t in tags:
            q = self._q.replace(self._placeholder, t)
            s = f'{q}{string}'
            strings.append(s)

        logger.debug(strings)

        outputs = super().generate(strings, **kwargs)
        entities = defaultdict(list)
        for no, t in enumerate(tags):
            e = outputs[no].split(' dan ')
            e = [e_ for e_ in e if len(e_) >= minimum_length and e_ != 'tiada' and e_ !=
                 'tiada jawapan' and e_ in string]
            e = list(set(e))
            entities[t].extend(e)

        return dict(entities)


class Aligment(Generator):
    def __init__(self, model, initial_text):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
        )

    def align(self):
        s = 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'
        input_ids = {'input_ids': tokenizer.encode(f'terjemah Inggeris ke Melayu: {s}', return_tensors='pt')}
        outputs = model.generate(**input_ids, max_length=256)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoder_input_ids = tokenizer.encode(outputs[0], return_tensors='pt')
        o = model.forward(**input_ids, decoder_input_ids=decoder_input_ids, output_attentions=True)
        c = []
        for a in o['cross_attentions']:
            c.append(a.detach().numpy())

        n = np.mean(np.mean(c, axis=0)[0], axis=0)
        s_t = tokenizer.tokenize(f'terjemah Inggeris ke Melayu: {s}')
        t_t = tokenizer.tokenize(outputs[0])
        rejected = tokenizer.all_special_tokens

        a = merge_sentencepiece_tokens_tagging(s_t, np.argmax(n.T, axis=1).tolist())

        prefix = 'terjemah Inggeris ke Melayu:'.split()
        f = []
        for no in range(len(a[0])):
            if a[0][no] not in prefix:
                f.append((a[0][no], t_t[a[1][no]]))


class ExtractiveQA(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            **kwargs,
        )

    def predict(
        self,
        paragraph_text: str,
        question_texts: List[str],
        **kwargs,
    ):
        """
        Predict Span from questions given a paragraph.

        Parameters
        ----------
        paragraph_text: str
        question_texts: List[str]
            List of questions, results really depends on case sensitive questions.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """

        text = remove_newlines(paragraph_text)
        strings = []
        for q in question_texts:
            s = f'konteks: {text} soalan: {remove_newlines(q)}'
            strings.append(s)

        return super().generate(strings, **kwargs)


class ExtractiveQAFlan(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            **kwargs,
        )

    def predict(
        self,
        paragraph_text: str,
        question_texts: List[str],
        **kwargs,
    ):
        """
        Predict Span from questions given a paragraph.

        Parameters
        ----------
        paragraph_text: str
        question_texts: List[str]
            List of questions, results really depends on case sensitive questions.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """

        text = remove_newlines(paragraph_text)
        strings = []
        for q in question_texts:
            s = f'read the following context and answer the question given: context: {text} question: {remove_newlines(q)}'
            strings.append(s)

        return super().generate(strings, **kwargs)


class Transformer(Base):
    def __init__(
        self,
        model,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)

        if self.tokenizer.slow_tokenizer_class in (RobertaTokenizer,):
            self._tokenizer_type = 'bpe'
            self._merge = merge_bpe_tokens
        elif self.tokenizer.slow_tokenizer_class in (ElectraTokenizer, BertTokenizer):
            self._tokenizer_type = 'wordpiece'
            self._merge = merge_wordpiece_tokens
        elif self.tokenizer.slow_tokenizer_class in (T5Tokenizer, AlbertTokenizer, XLNetTokenizer):
            self._tokenizer_type = 'sentencepiece'
            self._merge = merge_sentencepiece_tokens
        else:
            raise ValueError(
                'currently `malaya.transformer.load_huggingface` only supported `bpe`, `wordpiece` and `sentencepiece` tokenizer')

        self._t5 = False
        if 't5' in model:
            self.model = T5ForSequenceClassification.from_pretrained(model, **kwargs)
            self._t5 = True
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model, **kwargs)

    def forward(self, strings):
        cuda = next(self.model.parameters()).is_cuda
        input_ids = [{'input_ids': self.tokenizer.encode(s, return_tensors='pt')[0]} for s in strings]
        padded = self.tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)

        if self._t5:
            return self.model.embed(**padded, return_dict=True, output_attentions=True,
                                    output_hidden_states=True), padded
        else:
            return self.model(**padded, return_dict=True, output_attentions=True, output_hidden_states=True), padded

    def _method(self, layers, method, dim=0):
        method = method.lower()
        if method == 'last':
            layer = layers[-1]
        elif method == 'first':
            layer = layers[0]
        elif method == 'mean':
            layer = torch.mean(layers, dim=dim)
        else:
            raise ValueError('only supported `last`, `first` and `mean`.')
        return layer

    def vectorize(
        self,
        strings: List[str],
        method: str = 'last',
        method_token: str = 'first',
        t5_head_logits: bool = True,
        **kwargs,
    ):
        """
        Vectorize string inputs.

        Parameters
        ----------
        strings: List[str]
        method: str, optional (default='last')
            hidden layers supported. Allowed values:

            * ``'last'`` - last layer.
            * ``'first'`` - first layer.
            * ``'mean'`` - average all layers.

            This only applicable for non T5 models.
        method_token: str, optional (default='first')
            token layers supported. Allowed values:

            * ``'last'`` - last token.
            * ``'first'`` - first token.
            * ``'mean'`` - average all tokens.

            usually pretrained models trained on `first` token for classification task.
            This only applicable for non T5 models.
        t5_head_logits: str, optional (default=True)
            if True, will take head logits, else, last token.
            This only applicable for T5 models.

        Returns
        -------
        result: np.array
        """

        if self._t5:
            logits = self.forward(strings=strings)[0].logits
            if t5_head_logits:
                logits = logits[1]
            else:
                logits = logits[0]
            return to_numpy(logits)

        else:
            hidden_states = self.forward(strings=strings)[0].hidden_states
            stacked = torch.stack(hidden_states)
            layer = self._method(stacked, method)
            layer = layer.transpose(0, 1)
            return to_numpy(self._method(layer, method_token))

    def attention(
        self,
        strings: List[str],
        method: str = 'last',
        method_head: str = 'mean',
        t5_attention: str = 'cross_attentions',
        **kwargs,
    ):
        """
        Get attention string inputs.

        Parameters
        ----------
        strings: List[str]
        method: str, optional (default='last')
            Attention layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.
        method_head: str, optional (default='mean')
            attention head layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.
        t5_attention: str, optional (default='cross_attentions')
            attention type for T5 models. Allowed values:

            * ``'cross_attentions'`` - cross attention.
            * ``'encoder_attentions'`` - encoder attention.
            * ``'decoder_attentions'`` - decoder attention.

            This only applicable for T5 models.

        Returns
        -------
        result : List[List[Tuple[str, float]]]
        """

        forward = self.forward(strings=strings)
        if self._t5:
            attentions = getattr(forward[0], t5_attention)
        else:
            attentions = forward[0].attentions
        stacked = torch.stack(attentions)
        layer = self._method(stacked, method)
        layer = layer.transpose(0, 2).transpose(1, 2)
        cls_attn = to_numpy(self._method(layer, method_head))
        cls_attn = np.mean(cls_attn, axis=1)
        total_weights = np.sum(cls_attn, axis=-1, keepdims=True)
        attn = cls_attn / total_weights
        tokenized = [self.tokenizer.convert_ids_to_tokens(to_numpy(forward[1]['input_ids'][i]))
                     for i in range(len(forward[1]['input_ids']))]
        output = []
        for i in range(attn.shape[0]):
            output.append(
                self._merge(list(zip(tokenized[i], attn[i])),
                            rejected=self.tokenizer.all_special_tokens)
            )
        return output


class IsiPentingGenerator(Generator):
    def __init__(self, model, initial_text, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
            **kwargs,
        )

    @check_type
    def generate(
        self,
        strings: List[str],
        **kwargs,
    ):
        """
        generate a long text given a isi penting.

        Parameters
        ----------
        strings : List[str]
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        points = [
            f'{no + 1}. {remove_repeat_fullstop(string)}.'
            for no, string in enumerate(strings)
        ]
        points = ' '.join(points)
        points = f'karangan: {points}'
        results = super().generate([points], **kwargs)
        return [upperfirst(r) for r in results]


class Tatabahasa(Generator):
    def __init__(self, model, initial_text, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
            base_model=T5Tagging,
            **kwargs,
        )

    @check_type
    def generate(
        self,
        strings: List[str],
        **kwargs,
    ):
        """
        Fix kesalahan tatatabahasa.

        Parameters
        ----------
        strings : List[str]
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation
            Fix kesalahan tatabahasa supported all decoding methods except beam.

        Returns
        -------
        result: List[Tuple[str, int]]
        """
        if kwargs.get('num_beams', 0) > 0:
            raise ValueError('beam decoding is not supported.')

        outputs = super().generate(
            strings,
            output_attentions=True,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            return_generate=True,
            **kwargs,
        )
        last_layer = torch.stack([o[-1] for o in outputs.decoder_hidden_states])[:, :, 0]
        last_layer = last_layer.transpose(0, 1)
        tags = to_numpy(self.model.classification_head(last_layer)).argmax(axis=-1)
        results = []
        for no in range(len(outputs.sequences)):
            s = to_numpy(outputs.sequences[:, 1:][no])
            s = self.tokenizer.convert_ids_to_tokens(s)
            t = tags[no]
            merged = merge_sentencepiece_tokens_tagging(
                s, t, rejected=self.tokenizer.all_special_tokens
            )
            results.append(list(zip(merged[0], merged[1])))
        return results
