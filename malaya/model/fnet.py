import tensorflow as tf
import numpy as np
from malaya.function.activation import add_neutral as neutral
from malaya.function.activation import softmax
from malaya.text.bpe import bert_tokenization
from malaya.model.abstract import Classification, Abstract
from herpetologist import check_type
from typing import List


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


class FNet(Base):
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
        input_ids, input_masks, _, _ = bert_tokenization(
            self._tokenizer, strings
        )
        r = self._execute(
            inputs=[input_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1'],
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
        input_ids, input_masks, _, s_tokens = bert_tokenization(
            self._tokenizer, strings
        )
        r = self._execute(
            inputs=[input_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1'],
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


class MulticlassFNet(FNet, Classification):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
    ):
        FNet.__init__(
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


class BinaryFNet(FNet, Classification):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
    ):
        FNet.__init__(
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
    def predict(self, strings: List[str], add_neutral: bool = True):
        """
        classify list of strings.

        Parameters
        ----------
        strings: List[str]
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        result: List[str]
        """

        return self._predict(strings=strings, add_neutral=add_neutral)

    @check_type
    def predict_proba(self, strings: List[str], add_neutral: bool = True):
        """
        classify list of strings and return probability.

        Parameters
        ----------
        strings : List[str]
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        result: List[dict[str, float]]
        """

        return self._predict_proba(strings=strings, add_neutral=add_neutral)
