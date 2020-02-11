import tensorflow as tf
from ..texts._text_functions import (
    bert_tokenization_siamese,
    bert_tokenization,
    padding_sequence,
    merge_wordpiece_tokens,
    merge_sentencepiece_tokens,
    entities_textcleaning,
    merge_sentencepiece_tokens_tagging,
    merge_wordpiece_tokens_tagging,
    parse_bert_tagging,
    tag_chunk,
)
from .._utils._utils import add_neutral as neutral
from .._utils._parse_dependency import DependencyGraph
from .._utils._html import (
    _render_binary,
    _render_toxic,
    _render_emotion,
    _render_relevancy,
)
import numpy as np
from herpetologist import check_type
from typing import List


class BERT:
    def __init__(
        self,
        X,
        logits,
        segment_ids,
        input_masks,
        sess,
        tokenizer,
        cls,
        sep,
        label = ['negative', 'positive'],
    ):
        self._X = X
        self._logits = logits
        self._segment_ids = segment_ids
        self._input_masks = input_masks
        self._sess = sess
        self._tokenizer = tokenizer
        self._cls = cls
        self._sep = sep
        self._label = label


class BINARY_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        logits_seq,
        sess,
        tokenizer,
        cls,
        sep,
        attns,
        class_name,
        label = ['negative', 'positive'],
    ):
        BERT.__init__(
            self,
            X = X,
            segment_ids = segment_ids,
            input_masks = input_masks,
            logits = logits,
            sess = sess,
            tokenizer = tokenizer,
            cls = cls,
            sep = sep,
            label = label,
        )
        self._attns = attns
        self._logits_seq = logits_seq
        self._class_name = class_name
        self._softmax = tf.nn.softmax(self._logits)
        self._softmax_seq = tf.nn.softmax(self._logits_seq)

    def _predict(self, strings, add_neutral):
        input_ids, _, _, _ = bert_tokenization(
            self._tokenizer, strings, self._cls, self._sep
        )

        result = self._sess.run(self._softmax, feed_dict = {self._X: input_ids})
        if add_neutral:
            result = neutral(result)
        return result

    @check_type
    def predict(
        self, string: str, get_proba: bool = False, add_neutral: bool = True
    ):
        """
        classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        dictionary: results
        """

        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label
        result = self._predict([string], add_neutral = add_neutral)

        result = result[0]
        if get_proba:
            return {label[i]: result[i] for i in range(len(result))}
        else:
            return label[np.argmax(result)]

    @check_type
    def predict_batch(
        self,
        strings: List[str],
        get_proba: bool = False,
        add_neutral: bool = True,
    ):
        """
        classify list of strings.

        Parameters
        ----------
        strings : List[str]
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        list_dictionaries: list of results
        """

        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label

        results = self._predict(strings, add_neutral = add_neutral)

        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [label[result] for result in np.argmax(results, axis = 1)]

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
            raise Exception(
                "method not supported, only support 'last', 'first' and 'mean'"
            )
        label = self._label + ['neutral']

        batch_x, _, _, s_tokens = bert_tokenization(
            self._tokenizer, [string], cls = self._cls, sep = self._sep
        )
        result, attentions, words = self._sess.run(
            [self._softmax, self._attns, self._softmax_seq],
            feed_dict = {self._X: batch_x},
        )
        if method == 'first':
            cls_attn = list(attentions[0].values())[0][:, :, 0, :]

        if method == 'last':
            cls_attn = list(attentions[-1].values())[0][:, :, 0, :]

        if method == 'mean':
            combined_attentions = []
            for a in attentions:
                combined_attentions.append(list(a.values())[0])
            cls_attn = np.mean(combined_attentions, axis = 0).mean(axis = 2)

        cls_attn = np.mean(cls_attn, axis = 1)
        total_weights = np.sum(cls_attn, axis = -1, keepdims = True)
        attn = cls_attn / total_weights
        result = neutral(result)
        result = result[0]
        words = neutral(words[0])
        weights = []
        if '[' in self._cls:
            merged = merge_wordpiece_tokens(list(zip(s_tokens[0], attn[0])))
            for i in range(words.shape[1]):
                m = merge_wordpiece_tokens(
                    list(zip(s_tokens[0], words[:, i])), weighted = False
                )
                _, weight = zip(*m)
                weights.append(weight)
        else:
            merged = merge_sentencepiece_tokens(list(zip(s_tokens[0], attn[0])))
            for i in range(words.shape[1]):
                m = merge_sentencepiece_tokens(
                    list(zip(s_tokens[0], words[:, i])), weighted = False
                )
                _, weight = zip(*m)
                weights.append(weight)
        w, a = zip(*merged)
        words = np.array(weights).T
        distribution_words = words[:, np.argmax(words.sum(axis = 0))]
        y_histogram, x_histogram = np.histogram(
            distribution_words, bins = np.arange(0, 1, 0.05)
        )
        y_histogram = y_histogram / y_histogram.sum()
        x_attention = np.arange(len(w))
        left, right = np.unique(
            np.argmax(words, axis = 1), return_counts = True
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
        dict_result['class_name'] = self._class_name
        if visualization:
            _render_binary(dict_result)
        else:
            return dict_result


class MULTICLASS_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        logits_seq,
        sess,
        tokenizer,
        cls,
        sep,
        attns,
        class_name,
        label = ['negative', 'positive'],
    ):
        BERT.__init__(
            self,
            X = X,
            segment_ids = segment_ids,
            input_masks = input_masks,
            logits = logits,
            sess = sess,
            tokenizer = tokenizer,
            cls = cls,
            sep = sep,
            label = label,
        )
        self._attns = attns
        self._logits_seq = logits_seq
        self._class_name = class_name
        self._softmax = tf.nn.softmax(self._logits)
        self._softmax_seq = tf.nn.softmax(self._logits_seq)

    def _predict(self, strings):
        input_ids, _, _, _ = bert_tokenization(
            self._tokenizer, strings, self._cls, self._sep
        )

        result = self._sess.run(self._softmax, feed_dict = {self._X: input_ids})
        return result

    @check_type
    def predict(self, string: str, get_proba: bool = False):
        """
        classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        dictionary: results
        """

        result = self._predict([string])
        result = result[0]
        if get_proba:
            return {self._label[i]: result[i] for i in range(len(result))}
        else:
            return self._label[np.argmax(result)]

    @check_type
    def predict_batch(self, strings: List[str], get_proba: bool = False):
        """
        classify list of strings.

        Parameters
        ----------
        strings : List[str]
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        list_dictionaries: list of results
        """

        results = self._predict(strings)

        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {self._label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [
                self._label[result] for result in np.argmax(results, axis = 1)
            ]

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
            raise Exception(
                "method not supported, only support 'last', 'first' and 'mean'"
            )

        batch_x, _, _, s_tokens = bert_tokenization(
            self._tokenizer, [string], cls = self._cls, sep = self._sep
        )
        result, attentions, words = self._sess.run(
            [self._softmax, self._attns, self._softmax_seq],
            feed_dict = {self._X: batch_x},
        )
        if method == 'first':
            cls_attn = list(attentions[0].values())[0][:, :, 0, :]

        if method == 'last':
            cls_attn = list(attentions[-1].values())[0][:, :, 0, :]

        if method == 'mean':
            combined_attentions = []
            for a in attentions:
                combined_attentions.append(list(a.values())[0])
            cls_attn = np.mean(combined_attentions, axis = 0).mean(axis = 2)

        cls_attn = np.mean(cls_attn, axis = 1)
        total_weights = np.sum(cls_attn, axis = -1, keepdims = True)
        attn = cls_attn / total_weights
        result = result[0]
        words = words[0]
        weights = []
        if '[' in self._cls:
            merged = merge_wordpiece_tokens(list(zip(s_tokens[0], attn[0])))
            for i in range(words.shape[1]):
                m = merge_wordpiece_tokens(
                    list(zip(s_tokens[0], words[:, i])), weighted = False
                )
                _, weight = zip(*m)
                weights.append(weight)
        else:
            merged = merge_sentencepiece_tokens(list(zip(s_tokens[0], attn[0])))
            for i in range(words.shape[1]):
                m = merge_sentencepiece_tokens(
                    list(zip(s_tokens[0], words[:, i])), weighted = False
                )
                _, weight = zip(*m)
                weights.append(weight)
        w, a = zip(*merged)
        words = np.array(weights).T
        distribution_words = words[:, np.argmax(words.sum(axis = 0))]
        y_histogram, x_histogram = np.histogram(
            distribution_words, bins = np.arange(0, 1, 0.05)
        )
        y_histogram = y_histogram / y_histogram.sum()
        x_attention = np.arange(len(w))
        left, right = np.unique(
            np.argmax(words, axis = 1), return_counts = True
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
        dict_result['class_name'] = self._class_name
        if visualization:
            if self._class_name == 'relevancy':
                _render_relevancy(dict_result)
            else:
                _render_emotion(dict_result)
        else:
            return dict_result


class SIGMOID_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        logits_seq,
        sess,
        tokenizer,
        cls,
        sep,
        attns,
        class_name,
        label = ['negative', 'positive'],
    ):
        BERT.__init__(
            self,
            X = X,
            segment_ids = segment_ids,
            input_masks = input_masks,
            logits = logits,
            sess = sess,
            tokenizer = tokenizer,
            cls = cls,
            sep = sep,
            label = label,
        )
        self._attns = attns
        self._logits_seq = logits_seq
        self._class_name = class_name
        self._sigmoid = tf.nn.sigmoid(self._logits)
        self._sigmoid_seq = tf.nn.sigmoid(self._logits_seq)

    def _predict(self, strings):
        input_ids, _, _, _ = bert_tokenization(
            self._tokenizer, strings, self._cls, self._sep
        )

        result = self._sess.run(self._sigmoid, feed_dict = {self._X: input_ids})
        return result

    @check_type
    def predict(self, string: str, get_proba: bool = False):
        """
        classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        dictionary: results
        """

        result = self._predict([string])
        result = result[0]
        if get_proba:
            return {self._label[i]: result[i] for i in range(len(result))}
        else:
            probs = np.around(result)
            return [label for no, label in enumerate(self._label) if probs[no]]

    @check_type
    def predict_batch(self, strings: List[str], get_proba: bool = False):
        """
        classify list of strings.

        Parameters
        ----------
        strings : List[str]
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        list_dictionaries: list of results
        """

        probs = self._predict(strings)
        results = []
        if get_proba:
            for prob in probs:
                dict_result = {}
                for no, label in enumerate(self._label):
                    dict_result[label] = prob[no]
                results.append(dict_result)
        else:
            probs = np.around(probs)
            for prob in probs:
                list_result = []
                for no, label in enumerate(self._label):
                    if prob[no]:
                        list_result.append(label)
                results.append(list_result)

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
            raise Exception(
                "method not supported, only support 'last', 'first' and 'mean'"
            )

        batch_x, _, _, s_tokens = bert_tokenization(
            self._tokenizer, [string], cls = self._cls, sep = self._sep
        )
        result, attentions, words = self._sess.run(
            [self._sigmoid, self._attns, self._sigmoid_seq],
            feed_dict = {self._X: batch_x},
        )
        if method == 'first':
            cls_attn = list(attentions[0].values())[0][:, :, 0, :]

        if method == 'last':
            cls_attn = list(attentions[-1].values())[0][:, :, 0, :]

        if method == 'mean':
            combined_attentions = []
            for a in attentions:
                combined_attentions.append(list(a.values())[0])
            cls_attn = np.mean(combined_attentions, axis = 0).mean(axis = 2)

        cls_attn = np.mean(cls_attn, axis = 1)
        total_weights = np.sum(cls_attn, axis = -1, keepdims = True)
        attn = cls_attn / total_weights
        result = result[0]
        words = words[0]
        weights = []
        if '[' in self._cls:
            merged = merge_wordpiece_tokens(list(zip(s_tokens[0], attn[0])))
            for i in range(words.shape[1]):
                m = merge_wordpiece_tokens(
                    list(zip(s_tokens[0], words[:, i])), weighted = False
                )
                _, weight = zip(*m)
                weights.append(weight)
        else:
            merged = merge_sentencepiece_tokens(list(zip(s_tokens[0], attn[0])))
            for i in range(words.shape[1]):
                m = merge_sentencepiece_tokens(
                    list(zip(s_tokens[0], words[:, i])), weighted = False
                )
                _, weight = zip(*m)
                weights.append(weight)
        w, a = zip(*merged)
        words = np.array(weights).T
        distribution_words = words[:, np.argmax(words.sum(axis = 0))]
        y_histogram, x_histogram = np.histogram(
            distribution_words, bins = np.arange(0, 1, 0.05)
        )
        y_histogram = y_histogram / y_histogram.sum()
        x_attention = np.arange(len(w))
        left, right = np.unique(
            np.argmax(words, axis = 1), return_counts = True
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
        dict_result['class_name'] = self._class_name
        if visualization:
            _render_toxic(dict_result)
        else:
            return dict_result


class SIAMESE_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        sess,
        tokenizer,
        cls,
        sep,
        label = ['not similar', 'similar'],
    ):
        BERT.__init__(
            self,
            X = X,
            segment_ids = segment_ids,
            input_masks = input_masks,
            logits = logits,
            sess = sess,
            tokenizer = tokenizer,
            cls = cls,
            sep = sep,
            label = label,
        )
        self._softmax = tf.nn.softmax(self._logits)

    def _base(self, strings_left, strings_right):
        input_ids, input_masks, segment_ids = bert_tokenization_siamese(
            self._tokenizer,
            strings_left,
            strings_right,
            cls = self._cls,
            sep = self._sep,
        )

        return self._sess.run(
            self._softmax,
            feed_dict = {
                self._X: input_ids,
                self._segment_ids: segment_ids,
                self._input_masks: input_masks,
            },
        )

    @check_type
    def predict(self, string_left: str, string_right: str):
        """
        calculate similarity for two different texts.

        Parameters
        ----------
        string_left : str
        string_right : str

        Returns
        -------
        float: float
        """

        return self._base([string_left], [string_right])[0, 1]

    @check_type
    def predict_batch(self, strings_left: List[str], strings_right: List[str]):
        """
        calculate similarity for two different batch of texts.

        Parameters
        ----------
        string_left : List[str]
        string_right : List[str]

        Returns
        -------
        list: list of float
        """

        return self._base(strings_left, strings_right)[:, 1]


class TAGGING_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        sess,
        tokenizer,
        cls,
        sep,
        settings,
    ):
        BERT.__init__(
            self,
            X = X,
            segment_ids = segment_ids,
            input_masks = input_masks,
            logits = logits,
            sess = sess,
            tokenizer = tokenizer,
            cls = cls,
            sep = sep,
            label = None,
        )

        self._settings = settings
        self._settings['idx2tag'] = {
            int(k): v for k, v in self._settings['idx2tag'].items()
        }
        self._pos = 'organization' not in self._settings['tag2idx']

    @check_type
    def analyze(self, string: str):
        """
        Analyze a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: analyzed string
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
        string: tagged string
        """

        parsed_sequence, bert_sequence = parse_bert_tagging(
            string, self._tokenizer, self._cls, self._sep
        )
        predicted = self._sess.run(
            self._logits, feed_dict = {self._X: [parsed_sequence]}
        )[0]
        t = [self._settings['idx2tag'][d] for d in predicted]
        if '[' in self._cls:
            merged = merge_wordpiece_tokens_tagging(bert_sequence, t)
        else:
            merged = merge_sentencepiece_tokens_tagging(bert_sequence, t)
        return list(zip(merged[0], merged[1]))


class DEPENDENCY_BERT(BERT):
    def __init__(
        self,
        X,
        segment_ids,
        input_masks,
        logits,
        sess,
        tokenizer,
        cls,
        sep,
        settings,
        heads_seq,
    ):
        BERT.__init__(
            self,
            X = X,
            segment_ids = segment_ids,
            input_masks = input_masks,
            logits = logits,
            sess = sess,
            tokenizer = tokenizer,
            cls = cls,
            sep = sep,
            label = None,
        )

        self._tag2idx = settings
        self._idx2tag = {int(v): k for k, v in self._tag2idx.items()}
        self._heads_seq = heads_seq

    @check_type
    def predict(self, string: str):
        """
        Tag a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: tagged string
        """

        parsed_sequence, bert_sequence = parse_bert_tagging(
            string, self._tokenizer, self._cls, self._sep
        )
        tagging, depend = self._sess.run(
            [self._logits, self._heads_seq],
            feed_dict = {self._X: [parsed_sequence]},
        )
        tagging = [self._idx2tag[i] for i in tagging[0]]
        depend = depend[0] - 1

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
            indexing_.append((indexing[i][0], index))
            result.append(
                '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
                % (i + 1, tagging[i][0], index, tagging[i][1])
            )
        d = DependencyGraph('\n'.join(result), top_relation_label = 'root')
        return d, tagging, indexing_
