from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoTokenizer,
    RobertaTokenizer,
    ElectraTokenizer,
    BertTokenizer,
    T5Tokenizer,
    AlbertTokenizer,
    XLNetTokenizer,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from malaya.text.bpe import (
    merge_sentencepiece_tokens_tagging,
    merge_sentencepiece_tokens,
    merge_wordpiece_tokens,
    merge_bpe_tokens,
)
from malaya.text.function import (
    upperfirst,
    remove_repeat_fullstop,
    remove_newlines,
    remove_html_tags as f_remove_html_tags,
    pad_sentence_batch,
    tag_chunk,
    STOPWORDS,
)
from malaya_boilerplate.converter import ctranslate2_translator
from malaya.function.parse_dependency import DependencyGraph
from malaya.text.rouge import postprocess_summary, find_kata_encik
from malaya.torch_model.t5 import (
    T5ForSequenceClassification,
    T5ForTokenClassification,
    T5Tagging,
    T5Diaparser,
    T5Constituency,
    T5Embedding,
)
from malaya.torch_model.llama2 import LlamaModelEmbedding
from malaya.torch_model.constituency_modules import BatchIndices
from malaya_boilerplate.torch_utils import to_numpy
from malaya.function.activation import softmax
from malaya.parser.conll import CoNLL
from malaya.parser.alg import eisner, mst
from malaya.supervised.settings import dependency as dependency_settings
from malaya.graph.triplet import dict_to_list, rebel_format, parse_rebel
from collections import defaultdict
from typing import List, Callable, Dict
import numpy as np
import torch
import re
import logging

logger = logging.getLogger(__name__)

MAPPING_LANG = {'ms': 'Malay', 'en': 'Inggeris'}


class Base:
    def compile(self):
        if getattr(self, 'use_ctranslate2', False):
            raise ValueError('`compile` method not able to use for ctranslate2 model.')
        self.model = torch.compile(self.model)

    def eval(self, **kwargs):
        if getattr(self, 'use_ctranslate2', False):
            raise ValueError('`eval` method not able to use for ctranslate2 model.')
        return self.model.eval(**kwargs)

    def cuda(self, **kwargs):
        if getattr(self, 'use_ctranslate2', False):
            raise ValueError('`cuda` method not able to use for ctranslate2 model.')
        return self.model.cuda(**kwargs)

    def save_pretrained(self, *args, **kwargs):
        if getattr(self, 'use_ctranslate2', False):
            raise ValueError('`save_pretrained` method not able to use for ctranslate2 model.')
        return self.model.save_pretrained(
            *args, **kwargs), self.tokenizer.save_pretrained(*args, **kwargs)


class Generator(Base):
    def __init__(
        self,
        model,
        initial_text='',
        base_model=AutoModelForSeq2SeqLM,
        use_ctranslate2=False,
        **kwargs
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=False,
            **kwargs
        )
        self.is_gpt2tokenizer = 'GPT2Tokenizer' in str(type(self.tokenizer))
        self.use_ctranslate2 = use_ctranslate2

        if self.use_ctranslate2:
            if base_model != AutoModelForSeq2SeqLM:
                raise ValueError('`base_model` must `AutoModelForSeq2SeqLM` if `use_ctranslate2`.')

            self.model = ctranslate2_translator(model=model, **kwargs)
        else:
            self.model = base_model.from_pretrained(model, **kwargs)

        self._initial_text = initial_text

    def generate(self, strings: List[str], return_generate=False, prefix=None, **kwargs):
        """
        Generate texts from the input.

        Parameters
        ----------
        strings : List[str]
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

            If you are using `use_ctranslate2`, vector arguments pass to ctranslate2 `translate_batch` method.
            Read more at https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html?highlight=translate_batch#ctranslate2.Translator.translate_batch

        Returns
        -------
        result: List[str]
        """

        if isinstance(prefix, str):
            _initial_text = prefix
        else:
            _initial_text = self._initial_text

        logger.debug(f'generate, initial_text: {_initial_text}')
        logger.debug(f'generate, strings: {strings}')

        combined = []
        for s in strings:
            s = f'{_initial_text}{s}'
            if self.is_gpt2tokenizer:
                s += self.tokenizer.eos_token
            combined.append(s)

        if self.use_ctranslate2:
            tokens = [self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(s)) for s in combined]
            results = self.model.translate_batch(tokens, **kwargs)
            outputs = []
            for o in results:
                for h in o.hypotheses:
                    outputs.append(self.tokenizer.convert_tokens_to_ids(h))
        else:
            input_ids = [{'input_ids': self.tokenizer.encode(
                s, return_tensors='pt')[0]} for s in combined]
            padded = self.tokenizer.pad(input_ids, padding='longest', return_tensors='pt')
            for k in padded.keys():
                padded[k] = padded[k].to(self.model.device)
            padded.pop('token_type_ids', None)
            outputs = self.model.generate(**padded, **kwargs)

        if return_generate:
            return outputs
        else:
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def alignment(
        self,
        source: str,
        target: str,
    ):
        """
        align texts using cross attention and `dtw-python`.

        Parameters
        ----------
        source: List[str]
        target: List[str]

        Returns
        -------
        result: Dict
        """
        if self.use_ctranslate2:
            raise ValueError('`alignment` method not able to use for ctranslate2 model.')

        try:
            from dtw import dtw
        except Exception as e:
            raise ModuleNotFoundError(
                'dtw-python not installed. Please install it by `pip install dtw-python` and try again.'
            )

        input_ids = [{'input_ids': self.tokenizer.encode(
            f'{self._initial_text}{s}', return_tensors='pt')[0]} for s in source]

        padded = self.tokenizer.pad(input_ids, padding='longest')
        labels = self.tokenizer(target, padding=True, return_tensors='pt')['input_ids']
        padded['labels'] = labels
        for k in padded.keys():
            padded[k] = padded[k].to(self.model.device)

        with torch.no_grad():
            o = self.model(**padded, output_attentions=True, return_dict=True)

        weights = torch.cat(o['cross_attentions'])
        weights = weights.cpu()
        weights = torch.tensor(weights).softmax(dim=-1)
        w = weights / weights.norm(dim=-2, keepdim=True)
        matrix = w.mean(axis=(0, 1)).T
        alignment = dtw(np.ascontiguousarray(-matrix.double().numpy()))

        alignment_x = alignment.index2s
        alignment_y = alignment.index1s
        return {
            'alignment': matrix,
            'alignment_x': alignment_x,
            'alignment_y': alignment_y,
        }


class Prefix(Base):
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)

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
        padded = {'input_ids': self.tokenizer.encode(string, return_tensors='pt')}
        for k in padded.keys():
            padded[k] = padded[k].to(self.model.device)
        outputs = self.model.generate(**padded, **kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class Paraphrase(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='parafrasa: ',
            **kwargs,
        )

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
                s = find_kata_encik(strings[no], **kwargs)
                results[no] = s
        return results


class Summarization(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='ringkasan: ',
            **kwargs,
        )

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
                    strings[no // (len(results) // len(strings))],
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

        strings = []
        for i in range(len(strings_left)):
            s1 = strings_left[i]
            s2 = strings_right[i]
            s = f'ayat1: {s1} ayat2: {s2}'
            strings.append(s)

        input_ids = [{'input_ids': self.tokenizer.encode(
            s, return_tensors='pt')[0]} for s in strings]
        padded = self.tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = padded[k].to(self.model.device)
        padded.pop('token_type_ids', None)

        outputs = self.model(**padded, return_dict=True)
        return outputs

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
        prefix: str = 'ayat ini berkaitan tentang ',
        multilabel: bool = True,
    ):
        """
        classify list of strings and return probability.

        Parameters
        ----------
        strings: List[str]
        labels: List[str]
        prefix: str, optional (default='ayat ini berkaitan tentang ')
            prefix of labels to zero shot. Playing around with prefix can get better results.
        multilabel: bool, optional (default=True)
            probability of labels can be more than 1.0

        Returns
        -------
        list: List[Dict[str, float]]
        """
        strings_left, strings_right, mapping = [], [], defaultdict(list)
        index = 0
        for no, string in enumerate(strings):
            for label in labels:
                strings_left.append(string)
                text_label = f'{prefix}{label}'
                text_label = re.sub(r'[ ]+', ' ', text_label).strip()
                strings_right.append(text_label)
                mapping[no].append(index)
                index += 1

        outputs = super().forward(strings_left=strings_left, strings_right=strings_right)
        entail_contradiction_logits = outputs.logits[:, [0, 1]]
        if multilabel:
            probs = to_numpy(entail_contradiction_logits.softmax(dim=1)[:, 1])
        else:
            probs = to_numpy(entail_contradiction_logits[:, 1])

        results = []
        for k, v in mapping.items():
            if multilabel:
                result = {}
                for no, index in enumerate(v):
                    result[labels[no]] = probs[index]

            else:
                result = []
                for no, index in enumerate(v):
                    result.append(probs[index])
                p = softmax(result)

                result = {}
                for no, index in enumerate(v):
                    result[labels[no]] = p[no]

            results.append(result)
        return results


class ExtractiveQA(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            **kwargs,
        )
        self.flan_mode = 'flan' in model

    def predict(
        self,
        paragraph_text: str,
        question_texts: List[str],
        validate_answers: bool = True,
        validate_questions: bool = False,
        minimum_threshold_question: float = 0.05,
        **kwargs,
    ):
        """
        Predict extractive answers from questions given a paragraph.

        Parameters
        ----------
        paragraph_text: str
        question_texts: List[str]
            List of questions, results really depends on case sensitive questions.
        validate_answers: bool, optional (default=True)
            if True, will check the answer is inside the paragraph.
        validate_questions: bool, optional (default=False)
            if True, validate the question is subset of the paragraph using `sklearn.feature_extraction.text.CountVectorizer`
            it is only useful if `paragraph_text` and `question_texts` are the same language.
        minimum_threshold_question: float, optional (default=0.05)
            minimum score from `cosine_similarity`, only useful if `validate_questions = True`.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """

        text = remove_newlines(paragraph_text)
        strings, questions = [], []
        for q in question_texts:
            q_ = remove_newlines(q)
            if self.flan_mode:
                s = f'read the following context and answer the question given: context: {text} question: {q_}'
            else:
                s = f'ekstrak jawapan: {text} soalan: {q_}'
            strings.append(s)
            questions.append(q_)

        if validate_questions:
            tf = CountVectorizer(
                stop_words=STOPWORDS,
                token_pattern='[A-Za-z0-9\\-()]+',
                ngram_range=(1, 2)
            ).fit([text])
            v = tf.transform([text])
            scores = cosine_similarity(tf.transform(questions), v)[:, 0]
        else:
            scores = [1.0] * len(questions)

        r = super().generate(strings, **kwargs)
        if validate_answers:
            r = [r_.strip() if r_ in text else 'tiada jawapan' for r_ in r]

        results = []
        for no, r_ in enumerate(r):
            if scores[no // (len(r) // len(scores))] >= minimum_threshold_question:
                a = r_
            else:
                a = 'tiada jawapan'
            results.append(a)

        return results


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

        self.model = AutoModelForMaskedLM.from_pretrained(model, **kwargs)

    def forward(self, strings):
        input_ids = [{'input_ids': self.tokenizer.encode(
            s, return_tensors='pt')[0]} for s in strings]
        padded = self.tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = padded[k].to(self.model.device)

        return self.model(**padded, return_dict=True, output_attentions=True,
                          output_hidden_states=True), padded

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
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            **kwargs,
        )
        self._mode = [
            'surat-khabar',
            'tajuk-surat-khabar',
            'artikel',
            'penerangan-produk',
            'karangan',
        ]

    def generate(
        self,
        strings: List[str],
        mode: str = 'surat-khabar',
        remove_html_tags: bool = True,
        **kwargs,
    ):
        """
        generate a long text given a isi penting.

        Parameters
        ----------
        strings : List[str]
        mode: str, optional (default='surat-khabar')
            Mode supported. Allowed values:

            * ``'surat-khabar'`` - news style writing.
            * ``'tajuk-surat-khabar'`` - headline news style writing.
            * ``'artikel'`` - article style writing.
            * ``'penerangan-produk'`` - product description style writing.
            * ``'karangan'`` - karangan sekolah style writing.
        remove_html_tags: bool, optional (default=True)
            Will remove html tags using `malaya.text.function.remove_html_tags`.

        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        mode = mode.lower()
        if mode not in self._mode:
            raise ValueError(f'only supported one of {self._mode}')

        points = [
            f'{no + 1}. {remove_repeat_fullstop(string)}.'
            for no, string in enumerate(strings)
        ]
        points = ' '.join(points)
        points = f'{mode}: {points}'
        results = super().generate([points], **kwargs)
        results = [upperfirst(r) for r in results]
        if remove_html_tags:
            results = [f_remove_html_tags(r) for r in results]

        return results


class Tatabahasa(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='kesalahan tatabahasa:',
            base_model=T5Tagging,
            **kwargs,
        )

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


class Keyword(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            **kwargs,
        )

    def generate(
        self,
        strings: List[str],
        top_keywords: int = 5,
        **kwargs,
    ):
        """
        Generate texts from the input.

        Parameters
        ----------
        strings : List[str]
        top_keywords: int, optional (default=5)
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        prefix = f'{top_keywords} kata kunci: '
        strings = [f'{prefix}{s}' for s in strings]
        results = super().generate(strings, **kwargs)
        outputs = []
        for r in results:
            r = r.split(',')
            r = list(set(r))
            outputs.append(r)
        return outputs


class Constituency(Base):
    def __init__(self, model, **kwargs):
        kwargs.pop('initial_text', None)
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = T5Constituency.from_pretrained(model, **kwargs)
        self.START = '<s>'
        self.STOP = '</s>'
        self.TAG_UNK = 'UNK'

    def forward(self, string):
        all_input_ids = []
        all_word_start_mask = []
        all_word_end_mask = []

        string = [(None, w) for w in string.split()]
        sentences = [string]

        for snum, sentence in enumerate(sentences):

            tokens = []
            word_start_mask = []
            word_end_mask = []
            tokens.append(self.START)
            word_start_mask.append(1)
            word_end_mask.append(1)

            cleaned_words = []
            for _, word in sentence:
                cleaned_words.append(word)

            for word in cleaned_words:
                word_tokens = self.tokenizer.tokenize(word)
                for _ in range(len(word_tokens)):
                    word_start_mask.append(0)
                    word_end_mask.append(0)
                word_start_mask[len(tokens)] = 1
                word_end_mask[-1] = 1
                tokens.extend(word_tokens)
            tokens.append(self.STOP)
            word_start_mask.append(1)
            word_end_mask.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            all_input_ids.append(input_ids)
            all_word_start_mask.append(word_start_mask)
            all_word_end_mask.append(word_end_mask)

        padded = self.tokenizer.pad({
            'input_ids': all_input_ids,
        }, return_tensors='pt')
        all_word_start_mask = torch.from_numpy(
            np.array(pad_sentence_batch(all_word_start_mask, 0)[0]))
        all_word_end_mask = torch.from_numpy(np.array(pad_sentence_batch(all_word_end_mask, 0)[0]))

        padded['sentences'] = sentences
        padded['all_word_start_mask'] = all_word_start_mask
        padded['all_word_end_mask'] = all_word_end_mask

        packed_len = sum([(len(sentence) + 2) for sentence in sentences])
        i = 0
        tag_idxs = np.zeros(packed_len, dtype=int)
        batch_idxs = np.zeros(packed_len, dtype=int)
        for snum, sentence in enumerate(sentences):
            for (tag, word) in [(self.START, self.START)] + sentence + [(self.STOP, self.STOP)]:
                tag_idxs[i] = 0
                batch_idxs[i] = snum
                i += 1

        batch_idxs = BatchIndices(batch_idxs)
        padded['batch_idxs'] = batch_idxs
        tag_idxs = torch.from_numpy(tag_idxs)
        padded['tag_idxs'] = tag_idxs

        for k in padded.keys():
            if isinstance(padded[k], torch.Tensor):
                padded[k] = padded[k].to(self.model.device)

        padded['batch_idxs'].batch_idxs_torch = padded['batch_idxs'].batch_idxs_torch.to(
            self.model.device)

        return self.model(**padded)[0][0]

    def predict(self, string):
        """
        Parse a string into malaya.function.constituency.trees_newline.InternalParseNode.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: malaya.function.constituency.trees_newline.InternalParseNode object
        """
        return self.forward(string=string)


class Dependency(Base):
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = T5Diaparser.from_pretrained(model, **kwargs)

    def forward(self, string):

        texts, indices = [1], [0]
        text = string.split()
        for i in range(len(text)):
            t = self.tokenizer.encode(text[i], add_special_tokens=False)
            texts.extend(t)
            indices.extend([i + 1] * len(t))

        model_inputs = {
            'input_ids': texts,
            'attention_mask': [1] * len(texts),
            'indices': indices
        }

        padded = self.tokenizer.pad(
            [model_inputs],
            padding=True,
            max_length=None,
            pad_to_multiple_of=None,
            return_tensors='pt',
        )
        for k in padded.keys():
            padded[k] = padded[k].to(self.model.device)

        return self.model(**padded), padded

    def vectorize(self, string):
        return self.forward(string=string)[0].decoder_hidden_states

    def predict(
        self,
        string: str,
        validate_tree: bool = False,
        f_tree: Callable = eisner,
    ):
        """
        Tag a string. We assumed the string input been properly tokenized.

        Parameters
        ----------
        string: str
        validate_tree: bool, optional (default=False)
            validate arcs is a valid tree using `malaya.parser.conll.CoNLL.istree`.
            Originally from https://github.com/Unipisa/diaparser
        f_tree: Callable, optional (default=malaya.parser.alg.eisner)
            if arcs is not a tree, use approximate function to fix arcs.
            Originally from https://github.com/Unipisa/diaparser

        Returns
        -------
        result: Tuple
        """

        o, padded = self.forward(string=string)
        seq = padded['input_ids'][0, 1:]
        seq = self.tokenizer.convert_ids_to_tokens(seq)
        arc_preds = o.s_arc.argmax(axis=-1)
        rel_preds = o.s_rel.argmax(-1)

        if validate_tree:
            depend = to_numpy(arc_preds[0, 1:])
            indexing = merge_sentencepiece_tokens_tagging(
                seq,
                depend,
                rejected=self.tokenizer.all_special_tokens
            )
            if not CoNLL.istree(indexing[1]):
                s = to_numpy(o.s_arc[0])
                c = defaultdict(list)
                for i in range(len(s)):
                    c_ = defaultdict(list)
                    for k in range(len(s[i])):
                        c_[indices[k]].append(s[i][k])

                    for k in c_:
                        c_[k] = np.mean(c_[k])

                    c[indices[i]].append([v for v in c_.values()])

                new_score = np.zeros((len(c), len(c)))
                for k in c:
                    new_score[k] = np.mean(c[k], axis=0)

                new_index = f_tree(torch.Tensor(new_score).unsqueeze(0), torch.Tensor(
                    [0] + [1] * (len(new_score) - 1)).int().unsqueeze(0))[0].tolist()

                arcs = [0]
                for i in range(len(text)):
                    t = self.tokenizer.encode(text[i], add_special_tokens=False)
                    arcs.extend([new_index[i]] * len(t))

                arc_preds = torch.Tensor(arcs).long().unsqueeze(0)

        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        tagging = to_numpy(rel_preds[0, 1:])
        depend = to_numpy(arc_preds[0, 1:])
        tagging = [dependency_settings['idx2tag'][i] for i in tagging]

        tagging = merge_sentencepiece_tokens_tagging(
            seq, tagging, rejected=self.tokenizer.all_special_tokens)
        tagging = list(zip(*tagging))
        indexing = merge_sentencepiece_tokens_tagging(
            seq, depend, rejected=self.tokenizer.all_special_tokens)
        indexing = list(zip(*indexing))

        result, indexing_ = [], []
        for i in range(len(tagging)):
            index = int(indexing[i][1])
            if index > len(tagging):
                index = len(tagging)
            elif (i + 1) == index:
                index = index + 1
            elif index == -1:
                index = i
            indexing_.append((indexing[i][0], index))
            result.append(
                '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
                % (i + 1, tagging[i][0], index, tagging[i][1])
            )
        d = DependencyGraph('\n'.join(result), top_relation_label='root')
        return d, tagging, indexing_


class TexttoKG(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='teks ke grafik pengetahuan: ',
            **kwargs,
        )

    def generate(self, strings: List[Dict], got_networkx: bool = True, **kwargs):
        """
        Generate list of knowledge graphs from the input.

        Parameters
        ----------
        strings : List[str]
        got_networkx: bool, optional (default=True)
            If True, will generate networkx.MultiDiGraph.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[List[Dict]]
        """
        if got_networkx:
            try:
                import pandas as pd
                import networkx as nx
            except BaseException:
                logger.warning(
                    'pandas and networkx not installed. Please install it by `pip install pandas networkx` and try again. Will skip to generate networkx.MultiDiGraph'
                )
                got_networkx = False

        outputs_ = super().generate(strings, **kwargs)
        outputs = [parse_rebel(o) for o in outputs_]

        for no in range(len(outputs)):
            G = None
            if got_networkx:
                try:
                    df = pd.DataFrame(outputs[no])
                    G = nx.from_pandas_edgelist(
                        df,
                        source='head',
                        target='tail',
                        edge_attr='type',
                        create_using=nx.MultiDiGraph(),
                    )
                except Exception as e:
                    logger.warning(e)

            outputs[no] = {'G': G, 'triple': outputs[no], 'rebel': outputs_[no]}
        return outputs


class Translation(Generator):
    def __init__(self, model, from_lang=None, to_lang=None, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            **kwargs,
        )

        self.from_lang = from_lang
        self.to_lang = to_lang

        self.map_lang = {
            'en': 'Inggeris',
            'jav': 'Jawa',
            'bjn': 'Banjarese',
            'ms': 'Melayu',
            'ind': 'Indonesia',
            'pasar ms': 'pasar Melayu',
            'manglish': 'Manglish',
            'mandarin': 'Mandarin',
            'pasar mandarin': 'pasar Mandarin',
            'jawi': 'Jawi',
            'rumi': 'Rumi',
            'tamil': 'Tamil',
            'punjabi': 'Punjabi',
        }

        self.all_special_ids = [0, 1, 2]

    def generate(self, strings: List[str], to_lang: str = 'ms', **kwargs):
        """
        Generate texts from the input.

        Parameters
        ----------
        strings : List[str]
        to_lang: str, optional (default='ms')
            target language to translate.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

            If you are using `use_ctranslate2`, vector arguments pass to ctranslate2 `translate_batch` method.
            Read more at https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html?highlight=translate_batch#ctranslate2.Translator.translate_batch

        Returns
        -------
        result: List[str]
        """

        if to_lang not in self.to_lang:
            raise ValueError(f'this model does not support `{to_lang}` for `to_lang`')

        to_lang = self.map_lang[to_lang]

        prefix = f'terjemah ke {to_lang}: '
        if self.is_gpt2tokenizer:
            results = super().generate(strings, prefix=prefix, **kwargs)
        else:
            results = super().generate(strings, prefix=prefix, return_generate=True, **kwargs)
            results = self.tokenizer.batch_decode(
                [[i for i in o if i not in self.all_special_ids] for o in results],
                spaces_between_special_tokens=False,
            )
        return results


class Classification(Base):
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = T5ForSequenceClassification.from_pretrained(model, **kwargs)

    def forward(self, strings):
        padded = self.tokenizer(strings, padding='longest', return_tensors='pt')
        for k in padded.keys():
            padded[k] = padded[k].to(self.model.device)
        padded.pop('token_type_ids', None)

        return to_numpy(self.model(**padded)[0])

    def predict(self, strings):
        """
        classify list of strings.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """
        results = self.forward(strings=strings)
        argmax = np.argmax(results, axis=1)
        return [self.model.config.vocab[i] for i in argmax]

    def predict_proba(self, strings):
        """
        classify list of strings and return probability.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[dict[str, float]]
        """
        results = self.forward(strings=strings)
        results = softmax(results, axis=1)
        returns = []
        for r in results:
            returns.append({self.model.config.vocab[no]: float(r_) for no, r_ in enumerate(r)})
        return returns


class Tagging(Base):
    def __init__(self, model, **kwargs):

        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = T5ForTokenClassification.from_pretrained(model, **kwargs)

        self.rev_vocab = {v: k for k, v in self.model.config.vocab.items()}

    def forward(self, string: str):

        tokens = string.split()
        tokenized_inputs = self.tokenizer([tokens], truncation=True, is_split_into_words=True)
        tags = [[1] * len(t) for t in [tokens]]

        labels = []
        for i, label in enumerate(tags):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        indices = labels[0]
        padded = tokenized_inputs

        for k in padded.keys():
            padded[k] = torch.from_numpy(np.array(padded[k])).to(self.model.device)

        pred = self.model(**padded)[0]
        predictions = to_numpy(pred)[0].argmax(axis=1)
        filtered = [self.rev_vocab[int(predictions[i])]
                    for i in range(len(predictions)) if indices[i] != -100]
        filtered = [(tokens[i], filtered[i]) for i in range(len(filtered))]
        return filtered

    def predict(self, string: str):
        """
        Tag a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: Tuple[str, str]
        """

        return self.forward(string=string)

    def analyze(self, string: str):
        """
        Analyze a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: {'words': List[str], 'tags': [{'text': 'text', 'type': 'location', 'score': 1.0, 'beginOffset': 0, 'endOffset': 1}]}
        """
        predicted = self.predict(string)
        return tag_chunk(predicted)


class Embedding(Base):
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True, **kwargs)

    def encode(self, strings: List[str]):
        """
        Encode strings into embedding.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: np.array
        """
        padded = self.tokenizer(strings, return_tensors='pt', padding=True)
        for k in padded.keys():
            padded[k] = padded[k].to(self.model.device)

        padded.pop('token_type_ids', None)

        return to_numpy(self.model.encode(padded))


class Reranker(Base):
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, **kwargs)

    def sort(self, left_string: str, right_strings: List[str]):
        """
        Sort the strings.

        Parameters
        ----------
        left_string: str
            reference string.
        right_strings: List[str]
            query strings, list of strings need to sort based on reference string.

        Returns
        -------
        result: np.array
        """
        batch = []
        for s in right_strings:
            input_ids = self.tokenizer.encode_plus(left_string, s)
            input_ids.pop('token_type_ids')
            batch.append(input_ids)
        padded = self.tokenizer.pad(batch, return_tensors='pt')
        for k in padded.keys():
            padded[k] = padded[k].to(self.model.device)

        padded.pop('token_type_ids', None)

        return to_numpy(self.model(**padded).logits[:, 1])
