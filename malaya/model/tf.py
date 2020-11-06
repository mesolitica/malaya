import tensorflow as tf
import numpy as np
import re
from unidecode import unidecode
from malaya.text.function import (
    language_detection_textcleaning,
    split_into_sentences,
    transformer_textcleaning,
    translation_textcleaning,
    pad_sentence_batch,
    upperfirst,
)
from malaya.text.rouge import (
    filter_rouge,
    postprocessing_summarization,
    find_lapor_and_remove,
)
from malaya.text.bpe import (
    constituency_bert,
    constituency_xlnet,
    padding_sequence,
    PTB_TOKEN_ESCAPE,
    merge_sentencepiece_tokens,
)
from malaya.text import chart_decoder
from malaya.text.trees import tree_from_str
from herpetologist import check_type
from typing import List


def cleaning(string):
    return re.sub(r'[ ]+', ' ', string).strip()


def _convert_sparse_matrix_to_sparse_tensor(X, got_limit = False, limit = 5):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    if got_limit:
        coo.data[coo.data > limit] = limit
    return (
        tf.SparseTensorValue(indices, coo.col, coo.shape),
        tf.SparseTensorValue(indices, coo.data, coo.shape),
    )


class _LANG_MODEL:
    def __init__(self, dimension = 32, output = 6):
        self.X = tf.sparse_placeholder(tf.int32)
        self.W = tf.sparse_placeholder(tf.int32)
        self.Y = tf.placeholder(tf.int32, [None])
        embeddings = tf.Variable(tf.truncated_normal([400_000, dimension]))
        embed = tf.nn.embedding_lookup_sparse(
            embeddings, self.X, self.W, combiner = 'mean'
        )
        self.logits = tf.layers.dense(embed, output)


class DEEP_LANG:
    def __init__(self, path, vectorizer, label, bpe, type):
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._model = _LANG_MODEL()
            self._sess = tf.InteractiveSession()
            self._sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.trainable_variables())
            saver.restore(self._sess, path + '/model.ckpt')
        self._vectorizer = vectorizer
        self._label = label
        self._softmax = tf.nn.softmax(self._model.logits)
        self._bpe = bpe
        self._type = type

    def _classify(self, strings):
        strings = [language_detection_textcleaning(i) for i in strings]
        subs = [
            ' '.join(s)
            for s in self._bpe.encode(strings, output_type = self._type)
        ]
        transformed = self._vectorizer.transform(subs)
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        probs = self._sess.run(
            self._softmax,
            feed_dict = {self._model.X: batch_x[0], self._model.W: batch_x[1]},
        )
        return probs

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

        probs = self._classify(strings)
        dicts = []
        probs = np.argmax(probs, 1)
        for prob in probs:
            dicts.append(self._label[prob])
        return dicts

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
        dicts = []
        for i in range(probs.shape[0]):
            dicts.append({self._label[no]: k for no, k in enumerate(probs[i])})
        return dicts


class TRANSLATION:
    def __init__(self, X, greedy, beam, sess, encoder):

        self._X = X
        self._greedy = greedy
        self._beam = beam
        self._sess = sess
        self._encoder = encoder

    def _translate(self, strings, beam_search = True):
        encoded = [
            self._encoder.encode(translation_textcleaning(string)) + [1]
            for string in strings
        ]
        if beam_search:
            output = self._beam
        else:
            output = self._greedy
        batch_x = pad_sentence_batch(encoded, 0)[0]
        p = self._sess.run(output, feed_dict = {self._X: batch_x}).tolist()
        result = []
        for row in p:
            result.append(
                self._encoder.decode([i for i in row if i not in [0, 1]])
            )
        return result

    @check_type
    def translate(self, strings: List[str], beam_search: bool = True):
        """
        translate list of strings.

        Parameters
        ----------
        strings : List[str]
        beam_search : bool, (optional=True)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: List[str]
        """
        return self._translate(strings, beam_search = beam_search)


class CONSTITUENCY:
    def __init__(
        self,
        input_ids,
        word_end_mask,
        charts,
        tags,
        vectorizer,
        sess,
        tokenizer,
        dictionary,
        mode,
    ):

        self._input_ids = input_ids
        self._word_end_mask = word_end_mask
        self._charts = charts
        self._tags = tags
        self._vectorizer = vectorizer
        self._sess = sess
        self._tokenizer = tokenizer
        self._LABEL_VOCAB = dictionary['label']
        self._TAG_VOCAB = dictionary['tag']
        self._mode = mode

    def _parse(self, string):
        s = string.split()
        sentences = [s]
        if self._mode == 'bert':
            f = constituency_bert
        elif self._mode == 'xlnet':
            f = constituency_xlnet
        else:
            raise ValueError(
                'mode not supported, only supported `bert` or `xlnet`'
            )
        i, m, tokens = f(self._tokenizer, sentences)
        charts_val, tags_val = self._sess.run(
            (self._charts, self._tags),
            {self._input_ids: i, self._word_end_mask: m},
        )
        for snum, sentence in enumerate(sentences):
            chart_size = len(sentence) + 1
            chart = charts_val[snum, :chart_size, :chart_size, :]
        return s, tags_val[0], chart_decoder.decode(chart)

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
        s = string.split()
        sentences = [s]
        if self._mode == 'bert':
            f = constituency_bert
        elif self._mode == 'xlnet':
            f = constituency_xlnet
        else:
            raise ValueError(
                'mode not supported, only supported `bert` or `xlnet`'
            )
        i, m, tokens = f(self._tokenizer, sentences)
        v = self._sess.run(
            self._vectorizer, {self._input_ids: i, self._word_end_mask: m}
        )
        if self._mode == 'bert':
            v = v[0]
        elif self._mode == 'xlnet':
            v = v[:, 0]

        return merge_sentencepiece_tokens(
            list(zip(tokens[0], v[: len(tokens[0])])),
            weighted = False,
            vectorize = True,
            model = self._mode,
        )

    @check_type
    def parse_nltk_tree(self, string: str):

        """
        Parse a string into NLTK Tree, to make it useful, make sure you already installed tktinker.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: nltk.Tree object
        """

        try:
            import nltk
            from nltk import Tree
        except:
            raise ModuleNotFoundError(
                'nltk not installed. Please install it and try again.'
            )

        sentence, tags, (score, p_i, p_j, p_label) = self._parse(string)

        idx_cell = [-1]

        def make_tree():
            idx_cell[0] += 1
            idx = idx_cell[0]
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self._LABEL_VOCAB[label_idx]
            if (i + 1) >= j:
                word = sentence[i]
                tag = self._TAG_VOCAB[tags[i]]
                tag = PTB_TOKEN_ESCAPE.get(tag, tag)
                word = PTB_TOKEN_ESCAPE.get(word, word)
                tree = Tree(tag, [word])
                for sublabel in label[::-1]:
                    tree = Tree(sublabel, [tree])
                return [tree]
            else:
                left_trees = make_tree()
                right_trees = make_tree()
                children = left_trees + right_trees
                if label:
                    tree = Tree(label[-1], children)
                    for sublabel in reversed(label[:-1]):
                        tree = Tree(sublabel, [tree])
                    return [tree]
                else:
                    return children

        tree = make_tree()[0]
        tree.score = score
        return tree

    @check_type
    def parse_tree(self, string):

        """
        Parse a string into string treebank format.

        Parameters
        ----------
        string : str

        Returns
        -------
        result: malaya.text.trees.InternalTreebankNode class
        """

        sentence, tags, (score, p_i, p_j, p_label) = self._parse(string)

        idx_cell = [-1]

        def make_str():
            idx_cell[0] += 1
            idx = idx_cell[0]
            i, j, label_idx = p_i[idx], p_j[idx], p_label[idx]
            label = self._LABEL_VOCAB[label_idx]
            if (i + 1) >= j:
                word = sentence[i]
                tag = self._TAG_VOCAB[tags[i]]
                tag = PTB_TOKEN_ESCAPE.get(tag, tag)
                word = PTB_TOKEN_ESCAPE.get(word, word)
                s = '({} {})'.format(tag, word)
            else:
                children = []
                while (
                    (idx_cell[0] + 1) < len(p_i)
                    and i <= p_i[idx_cell[0] + 1]
                    and p_j[idx_cell[0] + 1] <= j
                ):
                    children.append(make_str())

                s = ' '.join(children)

            for sublabel in reversed(label):
                s = '({} {})'.format(sublabel, s)
            return s

        return tree_from_str(make_str())


class SUMMARIZATION:
    def __init__(self, X, top_p, greedy, beam, nucleus, sess, tokenizer):

        self._X = X
        self._top_p = top_p
        self._greedy = greedy
        self._beam = beam
        self._nucleus = nucleus
        self._sess = sess
        self._tokenizer = tokenizer
        self._mapping = {
            'greedy': self._greedy,
            'beam': self._beam,
            'nucleus': self._nucleus,
        }

    def _summarize(
        self,
        strings,
        mode,
        decoder = 'greedy',
        top_p = 0.7,
        postprocess = True,
        **kwargs,
    ):
        mode = mode.lower()
        if mode not in ['ringkasan', 'tajuk']:
            raise ValueError('mode only supports [`ringkasan`, `tajuk`]')

        if not 0 < top_p < 1:
            raise ValueError('top_p must be bigger than 0 and less than 1')

        decoder = decoder.lower()
        output = self._mapping.get(decoder)
        if not decoder:
            raise ValueError('mode only supports [`greedy`, `beam`, `nucleus`]')

        strings_ = [f'{mode}: {cleaning(string)}' for string in strings]

        batch_x = [self._tokenizer.encode(string) + [1] for string in strings_]
        batch_x = padding_sequence(batch_x)

        p = self._sess.run(
            output, feed_dict = {self._X: batch_x, self._top_p: top_p}
        ).tolist()

        results = []
        for no, r in enumerate(p):
            summary = self._tokenizer.decode(r)
            if postprocess:
                summary = filter_rouge(strings[no], summary, **kwargs)
                summary = postprocessing_summarization(summary)
                summary = find_lapor_and_remove(strings[no], summary)

            results.append(summary)

        return results

    @check_type
    def summarize(
        self,
        strings: List[str],
        mode: str = 'ringkasan',
        decoder: str = 'greedy',
        top_p: float = 0.7,
        postprocess: bool = True,
        **kwargs,
    ):
        """
        Summarize strings.

        Parameters
        ----------
        strings: List[str]
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.
        decoder: str
            mode for summarization decoder. Allowed values:

            * ``'greedy'`` - Beam width size 1, alpha 0.
            * ``'beam'`` - Beam width size 3, alpha 0.5 .
            * ``'nucleus'`` - Beam width size 1, with nucleus sampling.
        top_p: float, (default=0.7)
            cumulative distribution and cut off as soon as the CDF exceeds `top_p`.
            this is only useful if use `nucleus` decoder.
        postprocess: bool, optional (default=True)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """

        return self._summarize(
            strings = strings,
            mode = mode,
            decoder = decoder,
            top_p = top_p,
            postprocess = postprocess,
            **kwargs,
        )


class PARAPHRASE:
    def __init__(self, X, top_p, greedy, beam, nucleus, sess, tokenizer):

        self._X = X
        self._top_p = top_p
        self._greedy = greedy
        self._beam = beam
        self._nucleus = nucleus
        self._sess = sess
        self._tokenizer = tokenizer
        self._mapping = {
            'greedy': self._greedy,
            'beam': self._beam,
            'nucleus': self._nucleus,
        }

    def _paraphrase(self, strings, decoder = 'greedy', top_p = 0.7):

        if not 0 < top_p < 1:
            raise ValueError('top_p must be bigger than 0 and less than 1')

        decoder = decoder.lower()
        output = self._mapping.get(decoder)
        if not decoder:
            raise ValueError('mode only supports [`greedy`, `beam`, `nucleus`]')

        strings = [f'parafrasa: {cleaning(string)}' for string in strings]

        batch_x = [self._tokenizer.encode(string) + [1] for string in strings]
        batch_x = padding_sequence(batch_x)

        p = self._sess.run(
            output, feed_dict = {self._X: batch_x, self._top_p: top_p}
        ).tolist()

        results = [self._tokenizer.decode(r) for r in p]
        return results

    @check_type
    def paraphrase(
        self, strings: List[str], decoder: str = 'greedy', top_p: float = 0.7
    ):
        """
        Paraphrase strings.

        Parameters
        ----------
        strings: List[str]

        decoder: str
            mode for summarization decoder. Allowed values:

            * ``'greedy'`` - Beam width size 1, alpha 0.
            * ``'beam'`` - Beam width size 3, alpha 0.5 .
            * ``'nucleus'`` - Beam width size 1, with nucleus sampling.

        top_p: float, (default=0.7)
            cumulative distribution and cut off as soon as the CDF exceeds `top_p`.
            this is only useful if use `nucleus` decoder.

        Returns
        -------
        result: List[str]
        """

        return self._paraphrase(
            strings = strings, decoder = decoder, top_p = top_p
        )


class TRUE_CASE:
    def __init__(self, X, greedy, beam, sess, encoder):

        self._X = X
        self._greedy = greedy
        self._beam = beam
        self._sess = sess
        self._encoder = encoder

    def _true_case(self, strings, beam_search = True):
        encoded = self._encoder.encode(strings)
        if beam_search:
            output = self._beam
        else:
            output = self._greedy
        batch_x = pad_sentence_batch(encoded, 0)[0]
        p = self._sess.run(output, feed_dict = {self._X: batch_x}).tolist()
        result = self._encoder.decode(p)
        return result

    @check_type
    def true_case(self, strings: List[str], beam_search: bool = True):
        """
        True case strings.
        Example, "saya nak makan di us makanan di sana sedap" -> "Saya nak makan di US, makanan di sana sedap."

        Parameters
        ----------
        strings : List[str]
        beam_search : bool, (optional=True)
            If True, use beam search decoder, else use greedy decoder.

        Returns
        -------
        result: List[str]
        """
        return self._true_case(strings, beam_search = beam_search)
