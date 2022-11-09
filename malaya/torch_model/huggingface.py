from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from malaya.text.rouge import postprocess_summary, find_kata_encik
from malaya.text.function import remove_newlines
from malaya.torch_model.t5 import T5ForSequenceClassification
from malaya_boilerplate.torch_utils import to_tensor_cuda, to_numpy
from collections import defaultdict
from herpetologist import check_type
from typing import List
import re


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
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


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
        outputs = self.model.generate(**padded, **kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


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


class Similarity(Base):
    def __init__(self, model, use_fast_tokenizer=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast_tokenizer)
        self.model = T5ForSequenceClassification.from_pretrained(model)

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
    def __init__(self, model):
        Similarity.__init__(
            self,
            model=model,
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


class ZeroShotNER(Base):
    def __init__(self, model, use_fast_tokenizer=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=use_fast_tokenizer)
        self.model = T5ForSequenceClassification.from_pretrained(model)
        self._q = 'apakah entiti <entity>?'
        self._placeholder = '<entity>'

    def predict(self, string: str, tags: List[str], **kwargs):
        """
        classify entities in a string.

        Parameters
        ----------
        strings: str
            We assumed the string input been properly tokenized.
        tags: List[str]
        **kwargs: vector arguments pass to huggingface `generate` method.
            Read more at https://huggingface.co/docs/transformers/main_classes/text_generation

        Returns
        -------
        list: Dict[str, List[str]]
        """

        cuda = next(self.model.parameters()).is_cuda
        input_ids = []
        for t in tags:
            q = self._q.replace(self._placeholder, t)
            s = f'teks: {text} soalan: {q}'
            input_ids.append({'input_ids': tokenizer.encode(s, return_tensors='pt')[0]})

        padded = tokenizer.pad(input_ids, padding='longest')
        for k in padded.keys():
            padded[k] = to_tensor_cuda(padded[k], cuda)

        outputs = self.model.generate(**padded, **kwargs)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        entities = defaultdict(list)
        for no, t in enumerate(tags):
            e = outputs[no].split(' dan ')
            e = [e_ for e_ in e if len(e_) and e_ != 'tiada' and e_ != 'tiada jawapan' and e_ in string]
            e = list(set(e))
            entities[t].extend(e)

        return dict(entities)


class Aligment(Generator):
    def __init__(self, model, initial_text, use_fast_tokenizer=False):
        Generator.__init__(
            self,
            model=model,
            initial_text=initial_text,
            use_fast_tokenizer=use_fast_tokenizer,
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
    def __init__(self, model, use_fast_tokenizer=False):
        Generator.__init__(
            self,
            model=model,
            initial_text='',
            use_fast_tokenizer=use_fast_tokenizer,
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
