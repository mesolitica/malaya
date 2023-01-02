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
    summarization_textcleaning,
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
from malaya.function.activation import softmax
from malaya.parser.conll import CoNLL
from malaya.parser.alg import eisner, mst
from malaya.supervised.settings import dependency as dependency_settings
from collections import defaultdict
from herpetologist import check_type
from typing import List, Callable
import numpy as np
import torch
import re
import logging

logger = logging.getLogger(__name__)

MAPPING_LANG = {'ms': 'Malay', 'en': 'Inggeris'}


class Base:
    def cuda(self, **kwargs):
        return self.model.cuda(**kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs), self.tokenizer.save_pretrained(*args, **kwargs)


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

        logger.debug(f'generate, initial_text: {self._initial_text}')
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

    @check_type
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

    @check_type
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

    @check_type
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

    @check_type
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

                new_index = f_tree(torch.Tensor(new_score).unsqueeze(0),
                                   torch.Tensor([0] + [1] * (len(new_score) - 1)).int().unsqueeze(0))[0].tolist()

                arcs = [0]
                for i in range(len(text)):
                    t = self.tokenizer.encode(text[i], add_special_tokens=False)
                    arcs.extend([new_index[i]] * len(t))

                arc_preds = torch.Tensor(arcs).long().unsqueeze(0)

        rel_preds = rel_preds.gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        tagging = to_numpy(rel_preds[0, 1:])
        depend = to_numpy(arc_preds[0, 1:])
        tagging = [dependency_settings['idx2tag'][i] for i in tagging]

        tagging = merge_sentencepiece_tokens_tagging(seq, tagging, rejected=self.tokenizer.all_special_tokens)
        tagging = list(zip(*tagging))
        indexing = merge_sentencepiece_tokens_tagging(seq, depend, rejected=self.tokenizer.all_special_tokens)
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
