import tensorflow as tf
import numpy as np
from malaya.text.function import (
    language_detection_textcleaning,
    summarization_textcleaning,
    split_into_sentences,
    transformer_textcleaning,
    pad_sentence_batch,
    upperfirst,
)
from malaya.text.rouge import postprocess_summary
from malaya.text.bpe import (
    constituency_bert,
    constituency_xlnet,
    padding_sequence,
    PTB_TOKEN_ESCAPE,
    merge_sentencepiece_tokens,
    encode_pieces,
    merge_sentencepiece_tokens_tagging,
)
from malaya.text import chart_decoder
from malaya.text.trees import tree_from_str
from malaya.function.activation import softmax
from malaya.model.abstract import Seq2Seq, Classification, T2T, Abstract
from herpetologist import check_type
from typing import List


def _convert_sparse_matrix_to_sparse_tensor(X, got_limit = False, limit = 5):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col]).transpose()
    if got_limit:
        coo.data[coo.data > limit] = limit

    return coo.shape, coo.col, indices, coo.shape, coo.data, indices


class DeepLang(Classification):
    def __init__(
        self, input_nodes, output_nodes, sess, vectorizer, bpe, type, label
    ):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._vectorizer = vectorizer
        self._bpe = bpe
        self._type = type
        self._label = label

    def _classify(self, strings):
        strings = [language_detection_textcleaning(i) for i in strings]
        subs = [
            ' '.join(s)
            for s in self._bpe.encode(strings, output_type = self._type)
        ]
        transformed = self._vectorizer.transform(subs)
        batch_x = _convert_sparse_matrix_to_sparse_tensor(transformed)
        r = self._execute(
            inputs = batch_x,
            input_labels = [
                'X_Placeholder/shape',
                'X_Placeholder/values',
                'X_Placeholder/indices',
                'W_Placeholder/shape',
                'W_Placeholder/values',
                'W_Placeholder/indices',
            ],
            output_labels = ['logits'],
        )
        probs = softmax(r['logits'], axis = -1)
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


class Constituency(Abstract):
    def __init__(
        self, input_nodes, output_nodes, sess, tokenizer, dictionary, mode
    ):

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
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

        r = self._execute(
            inputs = [i, m],
            input_labels = ['input_ids', 'word_end_mask'],
            output_labels = ['charts', 'tags'],
        )
        charts_val, tags_val = r['charts'], r['tags']

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
        r = self._execute(
            inputs = [i, m],
            input_labels = ['input_ids', 'word_end_mask'],
            output_labels = ['vectorizer'],
        )
        v = r['vectorizer']
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


class Summarization(Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._tokenizer = tokenizer

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
        if decoder not in ['greedy', 'beam', 'nucleus']:
            raise ValueError('mode only supports [`greedy`, `beam`, `nucleus`]')

        strings_ = [
            f'{mode}: {summarization_textcleaning(string)}'
            for string in strings
        ]

        batch_x = [self._tokenizer.encode(string) + [1] for string in strings_]
        batch_x = padding_sequence(batch_x)

        r = self._execute(
            inputs = [batch_x, top_p],
            input_labels = ['Placeholder', 'Placeholder_2'],
            output_labels = [decoder],
        )
        p = r[decoder].tolist()

        results = []
        for no, r in enumerate(p):
            summary = self._tokenizer.decode(r)
            if postprocess and mode != 'tajuk':
                summary = postprocess_summary(strings[no], summary, **kwargs)

            results.append(summary)

        return results

    def greedy_decoder(
        self,
        strings: List[str],
        mode: str = 'ringkasan',
        postprocess: bool = True,
        **kwargs,
    ):
        """
        Summarize strings using greedy decoder.

        Parameters
        ----------
        strings: List[str]
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.
        postprocess: bool, optional (default=True)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """
        return self._summarize(
            strings = strings,
            mode = mode,
            decoder = 'greedy',
            top_p = 0.7,
            postprocess = postprocess,
            **kwargs,
        )

    def beam_decoder(
        self,
        strings: List[str],
        mode: str = 'ringkasan',
        postprocess: bool = True,
        **kwargs,
    ):
        """
        Summarize strings using beam decoder, beam width size 3, alpha 0.5 .

        Parameters
        ----------
        strings: List[str]
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.
        postprocess: bool, optional (default=True)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """
        return self._summarize(
            strings = strings,
            mode = mode,
            decoder = 'beam',
            top_p = 0.7,
            postprocess = postprocess,
            **kwargs,
        )

    def nucleus_decoder(
        self,
        strings: List[str],
        mode: str = 'ringkasan',
        top_p: float = 0.7,
        postprocess: bool = True,
        **kwargs,
    ):
        """
        Summarize strings using nucleus sampling.

        Parameters
        ----------
        strings: List[str]
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.
        top_p: float, (default=0.7)
            cumulative distribution and cut off as soon as the CDF exceeds `top_p`.
        postprocess: bool, optional (default=True)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """
        return self._summarize(
            strings = strings,
            mode = mode,
            decoder = 'nucleus',
            top_p = top_p,
            postprocess = postprocess,
            **kwargs,
        )


class Paraphrase(Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._tokenizer = tokenizer

    def _paraphrase(self, strings, decoder = 'greedy', top_p = 0.7):

        if not 0 < top_p < 1:
            raise ValueError('top_p must be bigger than 0 and less than 1')

        decoder = decoder.lower()
        if decoder not in ['greedy', 'beam', 'nucleus']:
            raise ValueError('mode only supports [`greedy`, `beam`, `nucleus`]')

        strings = [
            f'parafrasa: {summarization_textcleaning(string)}'
            for string in strings
        ]

        batch_x = [self._tokenizer.encode(string) + [1] for string in strings]
        batch_x = padding_sequence(batch_x)

        r = self._execute(
            inputs = [batch_x, top_p],
            input_labels = ['Placeholder', 'Placeholder_2'],
            output_labels = [decoder],
        )
        p = r[decoder].tolist()

        results = [self._tokenizer.decode(r) for r in p]
        return results

    def greedy_decoder(self, strings: List[str], **kwargs):
        """
        Paraphrase strings using greedy decoder.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """
        return self._paraphrase(
            strings = strings, decoder = 'greedy', top_p = 0.7, **kwargs
        )

    def beam_decoder(self, strings: List[str], **kwargs):
        """
        Paraphrase strings using beam decoder, beam width size 3, alpha 0.5 .

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """
        return self._paraphrase(
            strings = strings, decoder = 'beam', top_p = 0.7, **kwargs
        )

    def nucleus_decoder(self, strings: List[str], top_p: float = 0.7, **kwargs):
        """
        Paraphrase strings using nucleus sampling.

        Parameters
        ----------
        strings: List[str]
        top_p: float, (default=0.7)
            cumulative distribution and cut off as soon as the CDF exceeds `top_p`.

        Returns
        -------
        result: List[str]
        """
        return self._paraphrase(
            strings = strings, decoder = 'nucleus', top_p = top_p, **kwargs
        )


class Translation(T2T, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, encoder):

        T2T.__init__(
            self,
            input_nodes = input_nodes,
            output_nodes = output_nodes,
            sess = sess,
            encoder = encoder,
            translation_model = True,
        )

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
        return self._greedy_decoder(strings)

    def beam_decoder(self, strings: List[str]):
        """
        translate list of strings using beam decoder, beam width size 3, alpha 0.5 .

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        return self._beam_decoder(strings)


class TrueCase(T2T, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, encoder):
        T2T.__init__(
            self,
            input_nodes = input_nodes,
            output_nodes = output_nodes,
            sess = sess,
            encoder = encoder,
        )

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        True case strings using greedy decoder.
        Example, "saya nak makan di us makanan di sana sedap" -> "Saya nak makan di US, makanan di sana sedap."

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        return self._greedy_decoder(strings)

    @check_type
    def beam_decoder(self, strings: List[str]):
        """
        True case strings using beam decoder, beam width size 3, alpha 0.5 .
        Example, "saya nak makan di us makanan di sana sedap" -> "Saya nak makan di US, makanan di sana sedap."

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        return self._beam_decoder(strings)


class Segmentation(T2T, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, encoder):
        T2T.__init__(
            self,
            input_nodes = input_nodes,
            output_nodes = output_nodes,
            sess = sess,
            encoder = encoder,
        )

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        Segment strings using greedy decoder.
        Example, "sayasygkan negarasaya" -> "saya sygkan negara saya"

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        return self._greedy_decoder(strings)

    @check_type
    def beam_decoder(self, strings: List[str]):
        """
        Segment strings using beam decoder, beam width size 3, alpha 0.5 .
        Example, "sayasygkan negarasaya" -> "saya sygkan negara saya"

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        return self._beam_decoder(strings)


class Tatabahasa(Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._tokenizer = tokenizer

    def _predict(self, strings):
        sequences = [
            encode_pieces(
                self._tokenizer.sp,
                string,
                return_unicode = False,
                sample = False,
            )
            for string in strings
        ]
        batch_x = [self._tokenizer.encode(string) + [1] for string in strings]
        batch_x = padding_sequence(batch_x)
        r = self._execute(
            inputs = [batch_x],
            input_labels = ['x_placeholder'],
            output_labels = ['greedy', 'tag_greedy'],
        )
        p, tag = r['greedy'], r['tag_greedy']
        results = []
        nonzero = (p != 0).sum(axis = -1)
        for i in range(len(p)):
            r = self._tokenizer.decode(p[i].tolist())
            t = tag[i, : nonzero[i]]
            s = encode_pieces(
                self._tokenizer.sp, r, return_unicode = False, sample = False
            )
            merged = merge_sentencepiece_tokens_tagging(
                s + ['<cls>'], t, model = 'xlnet'
            )
            results.append(list(zip(merged[0], merged[1])))
        return results

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        Fix kesalahan tatatabahasa.

        Parameters
        ----------
        strings : List[str]

        Returns
        -------
        result: List[str]
        """
        return self._predict(strings)
