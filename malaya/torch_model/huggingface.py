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
    STOPWORDS,
)
from malaya.function.parse_dependency import DependencyGraph
from malaya.text.rouge import postprocess_summary, find_kata_encik
from malaya.torch_model.t5 import T5ForSequenceClassification, T5Tagging, T5Diaparser
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy
from malaya_boilerplate.converter import ctranslate2_translator
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
        if self.use_ctranslate2:
            raise ValueError('`compile` method not able to use for ctranslate2 model.')
        self.model = torch.compile(self.model)

    def eval(self, **kwargs):
        if self.use_ctranslate2:
            raise ValueError('`eval` method not able to use for ctranslate2 model.')
        return self.model.eval(**kwargs)

    def cuda(self, **kwargs):
        if self.use_ctranslate2:
            raise ValueError('`cuda` method not able to use for ctranslate2 model.')
        return self.model.cuda(**kwargs)

    def save_pretrained(self, *args, **kwargs):
        if self.use_ctranslate2:
            raise ValueError('`save_pretrained` method not able to use for ctranslate2 model.')
        return self.model.save_pretrained(
            *args, **kwargs), self.tokenizer.save_pretrained(*args, **kwargs)


class Generator(Base):
    def __init__(
        self,
        model,
        initial_text,
        base_model=AutoModelForSeq2SeqLM,
        use_ctranslate2=False,
        use_fast=True,
        **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast, **kwargs)
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

        if self.use_ctranslate2:
            tokens = [self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(f'{_initial_text}{s}')) for s in strings]
            results = self.model.translate_batch(tokens, **kwargs)
            outputs = []
            for o in results:
                for h in o.hypotheses:
                    outputs.append(self.tokenizer.convert_tokens_to_ids(h))
        else:
            cuda = next(self.model.parameters()).is_cuda
            input_ids = [{'input_ids': self.tokenizer.encode(
                f'{_initial_text}{s}', return_tensors='pt')[0]} for s in strings]
            padded = self.tokenizer.pad(input_ids, padding='longest', return_tensors='pt')
            for k in padded.keys():
                padded[k] = to_tensor_cuda(padded[k], cuda)
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

        cuda = next(self.model.parameters()).is_cuda
        input_ids = [{'input_ids': self.tokenizer.encode(
            f'{self._initial_text}{s}', return_tensors='pt')[0]} for s in source]

        padded = self.tokenizer.pad(input_ids, padding='longest')
        labels = self.tokenizer(target, padding=True, return_tensors='pt')['input_ids']
        padded['labels'] = labels
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)

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
        cuda = next(self.model.parameters()).is_cuda
        padded = {'input_ids': self.tokenizer.encode(string, return_tensors='pt')}
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
    def __init__(self, model, initial_text, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
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

        cuda = next(self.model.parameters()).is_cuda

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
            padded[k] = to_tensor_cuda(padded[k], cuda)

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
        prefix: str = 'ayat ini berkaitan tentang',
        multilabel: bool = True,
    ):
        """
        classify list of strings and return probability.

        Parameters
        ----------
        strings: List[str]
        labels: List[str]
        prefix: str, optional (default='ayat ini berkaitan tentang')
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
                text_label = f'{prefix} {label}'
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
        input_ids = {
            'input_ids': self.tokenizer.encode(
                f'terjemah Inggeris ke Melayu: {s}',
                return_tensors='pt')}
        outputs = self.model.generate(**input_ids, max_length=256)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoder_input_ids = self.tokenizer.encode(outputs[0], return_tensors='pt')
        o = self.model.forward(
            **input_ids,
            decoder_input_ids=decoder_input_ids,
            output_attentions=True)
        c = []
        for a in o['cross_attentions']:
            c.append(a.detach().numpy())

        n = np.mean(np.mean(c, axis=0)[0], axis=0)
        s_t = self.tokenizer.tokenize(f'terjemah Inggeris ke Melayu: {s}')
        t_t = self.tokenizer.tokenize(outputs[0])
        rejected = self.tokenizer.all_special_tokens

        a = merge_sentencepiece_tokens_tagging(s_t, np.argmax(n.T, axis=1).tolist())

        prefix = 'terjemah Inggeris ke Melayu:'.split()
        f = []
        for no in range(len(a[0])):
            if a[0][no] not in prefix:
                f.append((a[0][no], t_t[a[1][no]]))


class ExtractiveQA(Generator):
    def __init__(self, model, flan_mode=False, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            **kwargs,
        )
        self.flan_mode = flan_mode

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


class AbstractiveQA(Generator):
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
        langs: List[str],
        **kwargs,
    ):
        """
        Predict abstractive answers from questions given a paragraph.

        Parameters
        ----------
        paragraph_text: str
        question_texts: List[str]
            List of questions, results really depends on case sensitive questions.
        langs: List[str]
            Must same length as `question_texts`. Only accept `ms` or `en` only, case insensitive.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """

        if len(question_texts) != len(langs):
            raise ValueError('length of `langs` must be same as length of `question_texts`')

        langs_ = []

        for no, l in enumerate(langs):
            l_ = MAPPING_LANG.get(l.lower())
            if l_ is None:
                raise ValueError(f'langs[{no}] should only `ms` or `en`.')
            langs_.append(l_)

        text = remove_newlines(paragraph_text)
        strings = []
        for no, q in enumerate(question_texts):
            q_ = remove_newlines(q)
            s = f'abstrak jawapan {langs_[no]}: {text} soalan: {q_}'
            strings.append(s)

        r = super().generate(strings, **kwargs)
        return r


class GenerateQuestion(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            **kwargs,
        )

    def predict(
        self,
        paragraph_texts: List[str],
        answer_texts: List[str],
        langs: List[str],
        **kwargs,
    ):
        """
        Generate questions from answers given paragraphs.

        Parameters
        ----------
        paragraph_text: List[str]
        answer_texts: List[str]
            List of answers, can be extract or abstract.
        langs: List[str]
            Must same length as `answer_texts`. Only accept `ms` or `en` only, case insensitive.
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        if len(answer_texts) != len(langs):
            raise ValueError('length of `langs` must be same as length of `answer_texts`')

        langs_ = []

        for no, l in enumerate(langs):
            l_ = MAPPING_LANG.get(l.lower())
            if l_ is None:
                raise ValueError(f'langs[{no}] should only `ms` or `en`.')
            langs_.append(l_)

        strings = []
        for no, a in enumerate(answer_texts):
            a_ = remove_newlines(a)
            text = remove_newlines(paragraph_texts[no])
            s = f'bina soalan {langs_[no]}: jawapan: {a_}'
            strings.append(s)

        r = super().generate(strings, **kwargs)
        return r


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
        input_ids = [{'input_ids': self.tokenizer.encode(
            s, return_tensors='pt')[0]} for s in strings]
        padded = self.tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)

        if self._t5:
            return self.model.embed(**padded, return_dict=True, output_attentions=True,
                                    output_hidden_states=True), padded
        else:
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
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
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
    def __init__(self, model, initial_text, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
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


class Normalizer(Generator):
    def __init__(
        self,
        model,
        initial_text,
        normalizer,
        segmenter=None,
        text_scorer=None,
        **kwargs,
    ):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
            **kwargs,
        )
        self.normalizer = normalizer
        self.segmenter = segmenter
        self.text_scorer = text_scorer

    def generate(
        self,
        strings: List[str],
        **kwargs,
    ):
        """
        abstractive text normalization.

        Parameters
        ----------
        strings : List[str]
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

            Also vector arguments pass to `malaya.normalizer.rules.Normalizer.normalize`

        Returns
        -------
        result: List[str]
        """
        if self.normalizer is not None:
            for i in range(len(strings)):
                t = strings[i]
                try:
                    normalized = self.normalizer.normalize(
                        t, normalize_hingga=False, normalize_cardinal=False,
                        normalize_ordinal=False, normalize_pada_hari_bulan=False,
                        normalize_fraction=False, normalize_money=False, normalize_date=False,
                        normalize_time=False, normalize_ic=False, normalize_units=False,
                        normalize_url=False, normalize_percent=False, normalize_telephone=False,
                        text_scorer=self.text_scorer, segmenter=self.segmenter,
                        not_a_word_threshold=1e-9,
                    )['normalize']
                    logger.debug(f'input: {t}, normalized: {normalized}')
                    strings[i] = normalized
                except Exception as e:
                    logger.warning(f'input: {t}, exception {e}')
                    logger.warning(f'input: {t}, `self.normalizer` exception, skip to normalize.')

        return super().generate(strings, **kwargs)


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


class Dependency(Base):
    def __init__(self, model, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = T5Diaparser.from_pretrained(model, **kwargs)

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
        cuda = next(self.model.parameters()).is_cuda

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
            padded[k] = to_tensor_cuda(padded[k], cuda)

        o = self.model(**padded)

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
                    G = None
            else:
                G = None

            outputs[no] = {'G': G, 'triple': outputs[no], 'rebel': outputs_[no]}
        return outputs


class KGtoText(Generator):
    def __init__(self, model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='grafik pengetahuan ke teks: ',
            **kwargs,
        )

    def generate(self, kgs: List[List[Dict]], **kwargs):
        """
        Generate a text from list of knowledge graph dictionary.

        Parameters
        ----------
        kg: List[List[Dict]]
            list of list of {'head', 'type', 'tail'}
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        result: List[str]
        """
        for kg in kgs:
            for no, k in enumerate(kg):
                if 'head' not in k and 'type' not in k and 'tail' not in k:
                    raise ValueError('a dict must have `head`, `type` and `tail` properties.')
                elif not len(k['head']):
                    raise ValueError(f'`head` length must > 0 for knowledge graph index {no}')
                elif not len(k['type']):
                    raise ValueError(f'`head` length must > 0 for knowledge graph index {no}')
                elif not len(k['tail']):
                    raise ValueError(f'`head` length must > 0 for knowledge graph index {no}')

        rebels = [rebel_format(dict_to_list(kg)) for kg in kgs]
        return super().generate(rebels, **kwargs)


class Translation(Generator):
    def __init__(self, model, from_lang, to_lang, old_model, **kwargs):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            use_fast=True if old_model else False,
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
        }

        self.all_special_ids = [0, 1, 2]
        self.old_model = old_model

    def generate(self, strings: List[str], from_lang: str = None, to_lang: str = 'ms', **kwargs):
        """
        Generate texts from the input.

        Parameters
        ----------
        strings : List[str]
        from_lang: str, optional (default=None)
            old model required `from_lang` parameter to make it works properly,
            while new model not required.
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
        if self.old_model:
            if from_lang is None:
                raise ValueError('this model required `from_lang` parameter.')
            if from_lang not in self.from_lang:
                raise ValueError(f'this model does not support `{from_lang}` for `from_lang`')

        if to_lang not in self.to_lang:
            raise ValueError(f'this model does not support `{to_lang}` for `to_lang`')

        if from_lang is None:
            from_lang = ''
        else:
            from_lang = ' ' + self.map_lang[from_lang]

        to_lang = self.map_lang[to_lang]

        prefix = f'terjemah{from_lang} ke {to_lang}: '
        if self.old_model:
            results = super().generate(strings, prefix=prefix, **kwargs)
        else:
            results = super().generate(strings, prefix=prefix, return_generate=True, **kwargs)
            results = self.tokenizer.batch_decode(
                [[i for i in o if i not in self.all_special_ids] for o in results],
                spaces_between_special_tokens=False,
            )
        return results
