import tensorflow as tf
import numpy as np
from malaya.text.bpe import bert_tokenization


class Base:
    def __init__(
        self,
        X,
        logits,
        vectorizer,
        sess,
        tokenizer,
        label = ['negative', 'positive'],
    ):
        self._X = X
        self._logits = logits
        self._vectorizer = vectorizer
        self._sess = sess
        self._tokenizer = tokenizer
        self._label = label
        self._maxlen = 1536


class BigBird(Base):
    def __init__(
        self,
        X,
        logits,
        logits_seq,
        vectorizer,
        sess,
        tokenizer,
        class_name,
        label = ['negative', 'positive'],
    ):

        Base.__init__(
            self,
            X = X,
            logits = logits,
            vectorizer = vectorizer,
            sess = sess,
            tokenizer = tokenizer,
            label = label,
        )
        self._logits_seq = logits_seq
        self._class_name = class_name
        self._softmax = tf.nn.softmax(self._logits)
        self._softmax_seq = tf.nn.softmax(self._logits_seq)

    def _classify(self, strings):
        input_ids, _, _, _ = bert_tokenization(self._tokenizer, strings)
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(
            input_ids, padding = 'post', maxlen = self._maxlen
        )
        return self._sess.run(self._softmax, feed_dict = {self._X: input_ids})

    def _predict(self, strings, add_neutral = False):
        results = self._classify(strings)

        if add_neutral:
            result = neutral(results)
            label = self._label + ['neutral']
        else:
            label = self._label

        return [label[result] for result in np.argmax(results, axis = 1)]

    def _vectorize(self, strings, method = 'first'):
        method = method.lower()
        if method not in ['first', 'last', 'mean', 'word']:
            raise ValueError(
                "method not supported, only support 'first', 'last', 'mean' and 'word'"
            )
        input_ids, _, _, s_tokens = bert_tokenization(self._tokenizer, strings)
        v = self._sess.run(self._vectorizer, feed_dict = {self._X: input_ids})
        if method == 'first':
            v = v[:, 0]
        elif method == 'last':
            v = v[:, -1]
        elif method == 'mean':
            v = np.mean(v, axis = 1)
        else:
            v = [
                merge_sentencepiece_tokens(
                    list(zip(s_tokens[i], v[i][: len(s_tokens[i])])),
                    weighted = False,
                    vectorize = True,
                )
                for i in range(len(v))
            ]
        return v

    def _predict_proba(self, strings, add_neutral = False):
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


class MulticlassBigBird(Base):
    def __init__(self):
        pass
