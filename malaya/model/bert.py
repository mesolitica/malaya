import tensorflow as tf
from malaya.text.function import (
    entities_textcleaning,
    tag_chunk,
    split_into_sentences,
    translation_textcleaning,
)
from malaya.text.bpe import (
    bert_tokenization_siamese,
    bert_tokenization,
    padding_sequence,
    merge_wordpiece_tokens,
    merge_sentencepiece_tokens,
    merge_sentencepiece_tokens_tagging,
    parse_bert_tagging,
    parse_bert_token_tagging,
)
from malaya.function.activation import add_neutral as neutral
from malaya.function.activation import softmax, sigmoid
from malaya.function.parse_dependency import DependencyGraph
from malaya.function.html import (
    _render_binary,
    _render_toxic,
    _render_emotion,
    _render_relevancy,
)
from malaya.model.abstract import Classification, Seq2Seq, Tagging, Abstract
import numpy as np
from collections import defaultdict
from herpetologist import check_type
from typing import List, Tuple

render_dict = {
    'sentiment': _render_binary,
    'relevancy': _render_relevancy,
    'emotion': _render_emotion,
    'toxicity': _render_toxic,
    'subjectivity': _render_binary,
}


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


class BERT(Base):
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

    def _predict_words(
        self, string, method, visualization, add_neutral=False
    ):
        method = method.lower()
        if method not in ['last', 'first', 'mean']:
            raise ValueError(
                "method not supported, only support 'last', 'first' and 'mean'"
            )
        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label

        input_ids, input_masks, _, s_tokens = bert_tokenization(
            self._tokenizer, [string]
        )
        r = self._execute(
            inputs=[input_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits', 'attention', 'logits_seq'],
        )
        result = softmax(r['logits'], axis=-1)
        words = softmax(r['logits_seq'], axis=-1)
        attentions = r['attention']

        if method == 'first':
            cls_attn = list(attentions[0].values())[0][:, :, 0, :]

        if method == 'last':
            cls_attn = list(attentions[-1].values())[0][:, :, 0, :]

        if method == 'mean':
            combined_attentions = []
            for a in attentions:
                combined_attentions.append(list(a.values())[0])
            cls_attn = np.mean(combined_attentions, axis=0).mean(axis=2)

        cls_attn = np.mean(cls_attn, axis=1)
        total_weights = np.sum(cls_attn, axis=-1, keepdims=True)
        attn = cls_attn / total_weights
        words = words[0]

        if add_neutral:
            result = neutral(result)
            words = neutral(words)

        result = result[0]
        weights = []
        merged = merge_sentencepiece_tokens(list(zip(s_tokens[0], attn[0])))
        for i in range(words.shape[1]):
            m = merge_sentencepiece_tokens(
                list(zip(s_tokens[0], words[:, i])), weighted=False
            )
            _, weight = zip(*m)
            weights.append(weight)
        w, a = zip(*merged)
        words = np.array(weights).T
        distribution_words = words[:, np.argmax(words.sum(axis=0))]
        y_histogram, x_histogram = np.histogram(
            distribution_words, bins=np.arange(0, 1, 0.05)
        )
        y_histogram = y_histogram / y_histogram.sum()
        x_attention = np.arange(len(w))
        left, right = np.unique(
            np.argmax(words, axis=1), return_counts=True
        )
        left = left.tolist()
        y_barplot = []
        for i in range(len(label)):
            if i not in left:
                y_barplot.append(i)
            else:
                y_barplot.append(right[left.index(i)])

        dict_result = {label[i]: result[i] for i in range(len(result))}
        dict_result['alphas'] = {w: a[no] for no, w in enumerate(w)}
        dict_result['word'] = {w: words[no] for no, w in enumerate(w)}
        dict_result['histogram'] = {'x': x_histogram, 'y': y_histogram}
        dict_result['attention'] = {'x': x_attention, 'y': np.array(a)}
        dict_result['barplot'] = {'x': label, 'y': y_barplot}
        dict_result['module'] = self._module

        if visualization:
            render_dict[self._module](dict_result)
        else:
            return dict_result


class BinaryBERT(BERT, Classification):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
    ):
        BERT.__init__(
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

    @check_type
    def predict_words(
        self, string: str, method: str = 'last', visualization: bool = True
    ):
        """
        classify words.

        Parameters
        ----------
        string : str
        method : str, optional (default='last')
            Attention layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.
        visualization: bool, optional (default=True)
            If True, it will open the visualization dashboard.

        Returns
        -------
        result: dict
        """

        return self._predict_words(
            string=string,
            method=method,
            add_neutral=True,
            visualization=visualization,
        )


class MulticlassBERT(BERT, Classification):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        module,
        label=['negative', 'positive'],
    ):
        BERT.__init__(
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

    @check_type
    def predict_words(
        self, string: str, method: str = 'last', visualization: bool = True
    ):
        """
        classify words.

        Parameters
        ----------
        string : str
        method : str, optional (default='last')
            Attention layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.
        visualization: bool, optional (default=True)
            If True, it will open the visualization dashboard.

        Returns
        -------
        result: dict
        """
        return self._predict_words(
            string=string, method=method, visualization=visualization
        )


class SigmoidBERT(Base, Classification):
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
        return sigmoid(r['logits'])

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

    @check_type
    def predict(self, strings: List[str]):
        """
        classify list of strings.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[List[str]]
        """

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

        probs = self._classify(strings)
        results = []
        for prob in probs:
            dict_result = {}
            for no, label in enumerate(self._label):
                dict_result[label] = prob[no]
            results.append(dict_result)

        return results

    @check_type
    def predict_words(
        self, string: str, method: str = 'last', visualization: bool = True
    ):
        """
        classify words.

        Parameters
        ----------
        string : str
        method : str, optional (default='last')
            Attention layer supported. Allowed values:

            * ``'last'`` - attention from last layer.
            * ``'first'`` - attention from first layer.
            * ``'mean'`` - average attentions from all layers.
        visualization: bool, optional (default=True)
            If True, it will open the visualization dashboard.

        Returns
        -------
        dictionary: results
        """

        method = method.lower()
        if method not in ['last', 'first', 'mean']:
            raise ValueError(
                "method not supported, only support 'last', 'first' and 'mean'"
            )

        input_ids, input_masks, _, s_tokens = bert_tokenization(
            self._tokenizer, [string]
        )
        r = self._execute(
            inputs=[input_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits', 'attention', 'logits_seq'],
        )
        result = sigmoid(r['logits'])
        words = sigmoid(r['logits_seq'])
        attentions = r['attention']

        if method == 'first':
            cls_attn = list(attentions[0].values())[0][:, :, 0, :]

        if method == 'last':
            cls_attn = list(attentions[-1].values())[0][:, :, 0, :]

        if method == 'mean':
            combined_attentions = []
            for a in attentions:
                combined_attentions.append(list(a.values())[0])
            cls_attn = np.mean(combined_attentions, axis=0).mean(axis=2)

        cls_attn = np.mean(cls_attn, axis=1)
        total_weights = np.sum(cls_attn, axis=-1, keepdims=True)
        attn = cls_attn / total_weights
        result = result[0]
        words = words[0]
        weights = []
        merged = merge_sentencepiece_tokens(list(zip(s_tokens[0], attn[0])))
        for i in range(words.shape[1]):
            m = merge_sentencepiece_tokens(
                list(zip(s_tokens[0], words[:, i])), weighted=False
            )
            _, weight = zip(*m)
            weights.append(weight)
        w, a = zip(*merged)
        words = np.array(weights).T
        distribution_words = words[:, np.argmax(words.sum(axis=0))]
        y_histogram, x_histogram = np.histogram(
            distribution_words, bins=np.arange(0, 1, 0.05)
        )
        y_histogram = y_histogram / y_histogram.sum()
        x_attention = np.arange(len(w))
        left, right = np.unique(
            np.argmax(words, axis=1), return_counts=True
        )
        left = left.tolist()
        y_barplot = []
        for i in range(len(self._label)):
            if i not in left:
                y_barplot.append(i)
            else:
                y_barplot.append(right[left.index(i)])

        dict_result = {self._label[i]: result[i] for i in range(len(result))}
        dict_result['alphas'] = {w: a[no] for no, w in enumerate(w)}
        dict_result['word'] = {w: words[no] for no, w in enumerate(w)}
        dict_result['histogram'] = {'x': x_histogram, 'y': y_histogram}
        dict_result['attention'] = {'x': x_attention, 'y': np.array(a)}
        dict_result['barplot'] = {'x': self._label, 'y': y_barplot}
        dict_result['module'] = self._module
        if visualization:
            _render_toxic(dict_result)
        else:
            return dict_result


class SiameseBERT(Base):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        label=['not similar', 'similar'],
    ):
        Base.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer,
            label=label,
        )
        self._batch_size = 20

    def _base(self, strings_left, strings_right):
        input_ids, input_masks, segment_ids, _ = bert_tokenization_siamese(
            self._tokenizer, strings_left, strings_right
        )
        r = self._execute(
            inputs=[input_ids, segment_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1', 'Placeholder_2'],
            output_labels=['logits'],
        )
        return softmax(r['logits'], axis=-1)

    @check_type
    def vectorize(self, strings: List[str]):
        """
        Vectorize list of strings.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: np.array
        """
        input_ids, input_masks, segment_ids, _ = bert_tokenization(
            self._tokenizer, strings
        )
        segment_ids = np.array(segment_ids) + 1
        r = self._execute(
            inputs=[input_ids, segment_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1', 'Placeholder_2'],
            output_labels=['vectorizer'],
        )
        return r['vectorizer']

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
        list: list of float
        """

        if len(strings_left) != len(strings_right):
            raise ValueError(
                'length `strings_left` must be same as length `strings_right`'
            )

        return self._base(strings_left, strings_right)[:, 1]

    def _tree_plot(self, strings):
        l, r = [], []
        for s in strings:
            for s_ in strings:
                l.append(s)
                r.append(s_)

        results = []
        for i in range(0, len(l), self._batch_size):
            index = min(i + self._batch_size, len(l))
            x = l[i:index]
            y = r[i:index]
            results.append(self._base(x, y)[:, 1])

        results = np.concatenate(results, axis=0)
        results = np.reshape(results, (len(strings), len(strings)))
        return results

    @check_type
    def heatmap(
        self,
        strings: List[str],
        visualize: bool = True,
        annotate: bool = True,
        figsize: Tuple[int, int] = (7, 7),
    ):
        """
        plot a heatmap based on output from similarity

        Parameters
        ----------
        strings : list of str
            list of strings.
        visualize : bool
            if True, it will render plt.show, else return data.
        figsize : tuple, (default=(7, 7))
            figure size for plot.

        Returns
        -------
        result: list
            list of results
        """
        results = self._tree_plot(strings)

        if not visualize:
            return results
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set()
        except BaseException:
            raise ModuleNotFoundError(
                'matplotlib and seaborn not installed. Please install it and try again.'
            )

        plt.figure(figsize=figsize)
        g = sns.heatmap(
            results,
            cmap='Blues',
            xticklabels=strings,
            yticklabels=strings,
            annot=annotate,
        )
        plt.show()


class TaggingBERT(Base, Tagging):
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
            inputs=[[parsed_sequence], [input_mask]],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['vectorizer'],
        )
        v = r['vectorizer']
        v = v[0]
        return merge_sentencepiece_tokens(
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
            inputs=[[parsed_sequence], [input_mask]],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['logits'],
        )
        predicted = r['logits'][0]
        t = [self._settings['idx2tag'][d] for d in predicted]
        merged = merge_sentencepiece_tokens_tagging(bert_sequence, t)
        return list(zip(merged[0], merged[1]))


class DependencyBERT(Base):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer, settings, minus):
        Base.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer,
            label=None,
        )

        self._tag2idx = settings
        self._idx2tag = {int(v): k for k, v in self._tag2idx.items()}
        self._minus = minus

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
        parsed_sequence, input_mask, bert_sequence = parse_bert_tagging(
            string, self._tokenizer, space_after_punct=True
        )
        r = self._execute(
            inputs=[[parsed_sequence]],
            input_labels=['Placeholder'],
            output_labels=['vectorizer'],
        )
        v = r['vectorizer']
        v = v[0]
        return merge_sentencepiece_tokens(
            list(zip(bert_sequence, v[: len(bert_sequence)])),
            weighted=False,
            vectorize=True,
        )

    @check_type
    def predict(self, string: str):
        """
        Tag a string.

        Parameters
        ----------
        string: str

        Returns
        -------
        result: Tuple
        """

        parsed_sequence, input_mask, bert_sequence = parse_bert_tagging(
            string, self._tokenizer, space_after_punct=True
        )
        r = self._execute(
            inputs=[[parsed_sequence]],
            input_labels=['Placeholder'],
            output_labels=['logits', 'heads_seq'],
        )
        tagging, depend = r['logits'], r['heads_seq']
        tagging = [self._idx2tag[i] for i in tagging[0]]
        depend = depend[0] - self._minus

        for i in range(len(depend)):
            if depend[i] == 0 and tagging[i] != 'root':
                tagging[i] = 'root'
            elif depend[i] != 0 and tagging[i] == 'root':
                depend[i] = 0

        tagging = merge_sentencepiece_tokens_tagging(bert_sequence, tagging)
        tagging = list(zip(*tagging))
        indexing = merge_sentencepiece_tokens_tagging(bert_sequence, depend)
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


class ZeroshotBERT(Base):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        label=['not similar', 'similar'],
    ):
        Base.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer,
            label=label,
        )

    def _base(self, strings, labels):
        strings_left, strings_right, mapping = [], [], defaultdict(list)
        index = 0
        for no, string in enumerate(strings):
            for label in labels:
                strings_left.append(string)
                strings_right.append(f'teks ini adalah mengenai {label}')
                mapping[no].append(index)
                index += 1

        input_ids, input_masks, segment_ids, _ = bert_tokenization_siamese(
            self._tokenizer, strings_left, strings_right
        )

        r = self._execute(
            inputs=[input_ids, segment_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1', 'Placeholder_2'],
            output_labels=['logits'],
        )
        output = softmax(r['logits'], axis=-1)

        results = []
        for k, v in mapping.items():
            result = {}
            for no, index in enumerate(v):
                result[labels[no]] = output[index, 1]
            results.append(result)
        return results

    @check_type
    def vectorize(
        self, strings: List[str], labels: List[str], method: str = 'first'
    ):
        """
        vectorize a string.

        Parameters
        ----------
        strings: List[str]
        labels : List[str]
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
        strings_left, strings_right, combined = [], [], []
        for no, string in enumerate(strings):
            for label in labels:
                strings_left.append(string)
                strings_right.append(f'teks ini adalah mengenai {label}')
                combined.append((string, label))

        input_ids, input_masks, segment_ids, s_tokens = bert_tokenization_siamese(
            self._tokenizer, strings_left, strings_right
        )

        r = self._execute(
            inputs=[input_ids, segment_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1', 'Placeholder_2'],
            output_labels=['vectorizer'],
        )
        v = r['vectorizer']

        if len(v.shape) == 2:
            v = v.reshape((*np.array(input_ids).shape, -1))

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
        return combined, v

    @check_type
    def predict_proba(self, strings: List[str], labels: List[str]):
        """
        classify list of strings and return probability.

        Parameters
        ----------
        strings : List[str]
        labels : List[str]

        Returns
        -------
        list: list of float
        """

        if len(set(labels)) != len(labels):
            raise ValueError('labels must be unique.')

        return self._base(strings, labels)


class KeyphraseBERT(Base):
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        tokenizer,
        label=['not similar', 'similar'],
    ):
        Base.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer,
            label=label,
        )
        self._batch_size = 20

    def _base(self, strings_left, strings_right):
        input_ids_left, input_masks_left, _, _ = bert_tokenization(
            self._tokenizer, strings_left
        )
        input_ids_right, input_masks_right, _, _ = bert_tokenization(
            self._tokenizer, strings_right
        )
        r = self._execute(
            inputs=[
                input_ids_left,
                input_masks_left,
                input_ids_right,
                input_masks_right,
            ],
            input_labels=[
                'Placeholder',
                'Placeholder_1',
                'Placeholder_2',
                'Placeholder_3',
            ],
            output_labels=['logits'],
        )
        return softmax(r['logits'], axis=-1)

    @check_type
    def vectorize(self, strings: List[str]):
        """
        Vectorize list of strings.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: np.array
        """
        input_ids, input_masks, _, _ = bert_tokenization(
            self._tokenizer, strings
        )
        r = self._execute(
            inputs=[input_ids, input_masks],
            input_labels=['Placeholder', 'Placeholder_1'],
            output_labels=['bert/summary'],
        )
        return r['bert/summary']

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
        list: list of float
        """

        if len(strings_left) != len(strings_right):
            raise ValueError(
                'length `strings_left` must be same as length `strings_right`'
            )

        return self._base(strings_left, strings_right)[:, 1]
