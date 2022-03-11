import numpy as np
from malaya.text.function import tag_chunk
from malaya.function.activation import add_neutral as neutral
from malaya.function.activation import softmax, sigmoid
from malaya.text.bpe import (
    bert_tokenization,
    merge_wordpiece_tokens,
    merge_wordpiece_tokens_tagging,
    parse_bert_tagging,
    parse_bert_token_tagging,
)
from malaya.model.abstract import (
    Classification,
    Tagging,
    Base,
)
from herpetologist import check_type
from typing import List


class FastFormer(Base):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
        multilabels=False,
    ):

        Base.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer,
            label=label,
            module=module,
            multilabels=multilabels,
        )

    def _classify(self, strings):
        input_ids, input_masks, _, _ = bert_tokenization(
            self._tokenizer, strings, socialmedia=self._socialmedia
        )
        r = self._execute(
            inputs=[input_ids, input_masks],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        if self._multilabels:
            return sigmoid(r['logits'])
        else:
            return softmax(r['logits'], axis=-1)

    def _predict(self, strings, add_neutral=False):
        if self._multilabels:
            probs = self._classify(strings)
            results = []
            probs = np.around(probs)
            for prob in probs:
                list_result = []
                for no, label in enumerate(self._label):
                    if prob[no]:
                        list_result.append(label)
                results.append(list_result)

            return results
        else:
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
            self._tokenizer, strings,
        )
        r = self._execute(
            inputs=[input_ids, input_masks],
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
                merge_wordpiece_tokens(
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


class MulticlassFastFormer(FastFormer, Classification):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
    ):
        FastFormer.__init__(
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


class BinaryFastFormer(FastFormer, Classification):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
    ):
        FastFormer.__init__(
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


class SigmoidFastFormer(FastFormer, Classification):
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
            module=module,
            multilabels=True,
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


class TaggingFastFormer(Base, Tagging):
    def __init__(
        self, input_nodes, output_nodes, sess, tokenizer, settings, tok=None
    ):
        Base.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer,
            label=None,
        )

        self._settings = settings
        self._tok = tok
        self._settings['idx2tag'] = {
            int(k): v for k, v in self._settings['idx2tag'].items()
        }

    def _tokenize(self, string):
        if self._tok:
            parsed_sequence, input_mask, bert_sequence = parse_bert_token_tagging(
                string, self._tok, self._tokenizer
            )
        else:
            parsed_sequence, input_mask, bert_sequence = parse_bert_tagging(
                string, self._tokenizer, space_after_punct=True
            )
        return parsed_sequence, input_mask, bert_sequence

    @check_type
    def vectorize(self, string: str):
        """
        vectorize a string.

        Parameters
        ----------
        string: List[str]

        Returns
        -------
        result: np.array
        """
        parsed_sequence, input_mask, bert_sequence = self._tokenize(string)

        r = self._execute(
            inputs=[[parsed_sequence]],
            input_labels=['Placeholder'],
            output_labels=['vectorizer'],
        )
        v = r['vectorizer']
        v = v[0]
        return merge_wordpiece_tokens(
            list(zip(bert_sequence, v[: len(bert_sequence)])),
            weighted=False,
            vectorize=True,
        )

    @check_type
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

    @check_type
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
        parsed_sequence, input_mask, bert_sequence = self._tokenize(string)
        r = self._execute(
            inputs=[[parsed_sequence]],
            input_labels=['Placeholder'],
            output_labels=['logits'],
        )
        predicted = r['logits'][0]
        t = [self._settings['idx2tag'][d] for d in predicted]
        merged = merge_wordpiece_tokens_tagging(bert_sequence, t)
        return list(zip(merged[0], merged[1]))
