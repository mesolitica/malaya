import tensorflow as tf
import numpy as np
from malaya.function.activation import add_neutral as neutral
from malaya.function.activation import softmax, sigmoid
from malaya.text.function import (
    translation_textcleaning,
    summarization_textcleaning,
)
from malaya.text.rouge import postprocess_summary
from malaya.text.bpe import bert_tokenization
from malaya.model.abstract import Classification, Seq2Seq, Abstract
from herpetologist import check_type
from typing import List

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences


class Base(Abstract):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        label=['negative', 'positive'],
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._tokenizer = tokenizer
        self._label = label
        self._maxlen = 1024


class BigBird(Base):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
    ):

        Base.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer,
            label=label,
        )
        self._module = module

    def _classify(self, strings):
        input_ids, _, _, _ = bert_tokenization(self._tokenizer, strings)
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(
            input_ids, padding='post', maxlen=self._maxlen
        )
        r = self._execute(
            inputs=[input_ids],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        return softmax(r['logits'], axis=-1)

    def _predict(self, strings, add_neutral=False):
        results = self._classify(strings)

        if add_neutral:
            result = neutral(results)
            label = self._label + ['neutral']
        else:
            label = self._label

        return [label[result] for result in np.argmax(results, axis=1)]

    def _vectorize(self, strings, method='first'):
        method = method.lower()
        if method not in ['first', 'last', 'mean', 'word']:
            raise ValueError(
                "method not supported, only support 'first', 'last', 'mean' and 'word'"
            )
        input_ids, _, _, s_tokens = bert_tokenization(self._tokenizer, strings)
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(
            input_ids, padding='post', maxlen=self._maxlen
        )
        r = self._execute(
            inputs=[input_ids],
            input_labels=['Placeholder'],
            output_labels=['vectorizer'],
        )
        v = r['vectorizer']
        if method == 'first':
            v = v[:, 0]
        elif method == 'last':
            v = v[:, -1]
        elif method == 'mean':
            v = np.mean(v, axis=1)
        else:
            v = [
                merge_sentencepiece_tokens(
                    list(zip(s_tokens[i], v[i][: len(s_tokens[i])])),
                    weighted=False,
                    vectorize=True,
                )
                for i in range(len(v))
            ]
        return v

    def _predict_proba(self, strings, add_neutral=False):
        results = self._classify(strings)

        if add_neutral:
            results = neutral(results)
            label = self._label + ['neutral']
        else:
            label = self._label

        outputs = []
        for result in results:
            outputs.append({label[i]: result[i] for i in range(len(result))})
        return outputs


class MulticlassBigBird(BigBird, Classification):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
    ):
        BigBird.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer,
            module=module,
            label=label,
        )

    @check_type
    def vectorize(self, strings: List[str], method: str = 'first'):
        """
        vectorize list of strings.

        Parameters
        ----------
        strings: List[str]
        method : str, optional (default='first')
            Vectorization layer supported. Allowed values:

            * ``'last'`` - vector from last sequence.
            * ``'first'`` - vector from first sequence.
            * ``'mean'`` - average vectors from all sequences.
            * ``'word'`` - average vectors based on tokens.

        Returns
        -------
        result: np.array
        """

        return self._vectorize(strings=strings, method=method)

    @check_type
    def predict(self, strings: List[str]):
        """
        classify list of strings.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """

        return self._predict(strings=strings)

    @check_type
    def predict_proba(self, strings: List[str]):
        """
        classify list of strings and return probability.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[dict[str, float]]
        """

        return self._predict_proba(strings=strings)


class Translation(Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, encoder, maxlen):

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._encoder = encoder
        self._maxlen = maxlen

    def _translate(self, strings):
        encoded = [
            self._encoder.encode(translation_textcleaning(string)) + [1]
            for string in strings
        ]
        batch_x = pad_sequences(
            encoded, padding='post', maxlen=self._maxlen
        )
        r = self._execute(
            inputs=[batch_x],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        p = r['logits']
        result = []
        for r in p:
            result.append(
                self._encoder.decode([i for i in r.tolist() if i > 0])
            )
        return result

    def greedy_decoder(self, strings: List[str]):
        """
        translate list of strings.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        return self._translate(strings)


class Summarization(Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer, maxlen):

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._tokenizer = tokenizer
        self._maxlen = maxlen

    def _summarize(
        self,
        strings,
        top_p=0.7,
        temperature=1.0,
        postprocess=True,
        **kwargs,
    ):
        strings_ = [summarization_textcleaning(string) for string in strings]
        batch_x = [self._tokenizer.encode(string) + [1] for string in strings_]
        batch_x = pad_sequences(
            batch_x, padding='post', maxlen=self._maxlen
        )

        r = self._execute(
            inputs=[batch_x, top_p, temperature],
            input_labels=['Placeholder', 'top_p', 'temperature'],
            output_labels=['logits'],
        )
        p = r['logits'].tolist()

        results = []
        for no, r in enumerate(p):
            summary = self._tokenizer.decode(r)
            if postprocess:
                summary = postprocess_summary(strings[no], summary, **kwargs)

            results.append(summary)

        return results

    @check_type
    def greedy_decoder(
        self,
        strings: List[str],
        temperature: float = 0.0,
        postprocess: bool = False,
        **kwargs,
    ):
        """
        Summarize strings using greedy decoder.

        Parameters
        ----------
        strings: List[str]
        temperature: float, (default=0.0)
            logits * -log(random.uniform) * temperature.
        postprocess: bool, optional (default=False)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """
        return self._summarize(
            strings=strings,
            top_p=0.0,
            temperature=temperature,
            postprocess=postprocess,
            **kwargs,
        )

    @check_type
    def nucleus_decoder(
        self,
        strings: List[str],
        top_p: float = 0.7,
        temperature: float = 0.1,
        postprocess: bool = False,
        **kwargs,
    ):
        """
        Summarize strings using nucleus decoder.

        Parameters
        ----------
        strings: List[str]
        top_p: float, (default=0.7)
            cumulative distribution and cut off as soon as the CDF exceeds `top_p`.
        temperature: float, (default=0.3)
            logits * -log(random.uniform) * temperature.
        postprocess: bool, optional (default=False)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """
        return self._summarize(
            strings=strings,
            top_p=top_p,
            temperature=temperature,
            postprocess=postprocess,
            **kwargs,
        )
