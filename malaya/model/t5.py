import tensorflow as tf
from malaya.text.function import (
    transformer_textcleaning,
    summarization_textcleaning,
    split_into_sentences,
    upperfirst,
)
from malaya.text.rouge import postprocess_summary
from malaya.text.knowledge_graph import parse_triples
from malaya.model.abstract import Seq2Seq, Abstract
from herpetologist import check_type
from typing import List


def remove_repeat_fullstop(string):
    return ' '.join([k.strip() for k in string.split('.') if len(k.strip())])


class T5(Abstract):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._tokenizer = tokenizer

    def _predict(self, strings):
        r = self._execute(
            inputs=[strings],
            input_labels=['inputs'],
            output_labels=['decode'],
        )
        return self._tokenizer.decode(r['decode'].tolist())


class Summarization(T5, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        T5.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer
        )

    def _summarize(self, strings, mode, postprocess, **kwargs):
        summaries = self._predict([f'ringkasan: {summarization_textcleaning(string)}' for string in strings])
        if postprocess and mode != 'tajuk':
            summaries = [postprocess_summary(strings[no], summary, **kwargs) for no, summary in enumerate(summaries)]
        return summaries

    @check_type
    def greedy_decoder(
        self,
        strings: List[str],
        postprocess: bool = False,
        **kwargs,
    ):
        """
        Summarize strings.

        Parameters
        ----------
        strings: List[str]
        postprocess: bool, optional (default=False)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """

        return self._summarize(strings, mode, postprocess, **kwargs)


class Generator(T5, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        T5.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer
        )

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        generate a long text given a isi penting.
        Decoder is greedy decoder with beam width size 1, alpha 0.5 .

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: str
        """

        points = [
            f'{no + 1}. {remove_repeat_fullstop(string)}.'
            for no, string in enumerate(strings)
        ]
        points = ' '.join(points)
        points = f'karangan: {points}'
        return upperfirst(self._predict([summarization_textcleaning(points)])[0])


class Paraphrase(T5, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        T5.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer
        )

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        paraphrase strings.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """
        paraphrases = self._predict([f'parafrasa: {summarization_textcleaning(string)}' for string in strings])
        return [upperfirst(paraphrase) for paraphrase in paraphrases]


class KnowledgeGraph(T5, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        T5.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer
        )

    @check_type
    def greedy_decoder(
        self,
        strings: List[str],
        get_networkx: bool = True,
    ):
        """
        Generate triples knowledge graph using greedy decoder.
        Example, "Joseph Enanga juga bermain untuk Union Douala." -> "Joseph Enanga member of sports team Union Douala"

        Parameters
        ----------
        strings: List[str]
        get_networkx: bool, optional (default=True)
            If True, will generate networkx.MultiDiGraph.

        Returns
        -------
        result: List[Dict]
        """
        if get_networkx:
            try:
                import pandas as pd
                import networkx as nx
            except BaseException:
                logging.warning(
                    'pandas and networkx not installed. Please install it by `pip install pandas networkx` and try again. Will skip to generate networkx.MultiDiGraph'
                )
                get_networkx = False

        results = self._predict([f'grafik pengetahuan: {summarization_textcleaning(string)}' for string in strings])

        outputs = []
        for result in results:
            r, last_object = parse_triples(result)
            o = {'result': r, 'main_object': last_object, 'triple': result}
            if get_networkx and len(r):
                df = pd.DataFrame(r)
                G = nx.from_pandas_edgelist(
                    df,
                    source='subject',
                    target='object',
                    edge_attr='relation',
                    create_using=nx.MultiDiGraph(),
                )
                o['G'] = G
            outputs.append(o)

        return outputs


class Spell(T5, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        T5.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer
        )
        from malaya.preprocessing import Tokenizer
        self._word_tokenizer = Tokenizer(duration=False, date=False).tokenize

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        spelling correction for strings.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """

        return self._predict([f"betulkan ejaan: {' '.join(self._word_tokenizer(string))}" for string in strings])


class Segmentation(T5, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        T5.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer
        )

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        text segmentation.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """

        return self._predict([f'segmentasi: {string}' for string in strings])


class CommonGen(T5, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        T5.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer
        )

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        text generator given keywords.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """

        return self._predict([f'kata kunci: {string}' for string in strings])


class TrueCase(T5, Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):
        T5.__init__(
            self,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            sess=sess,
            tokenizer=tokenizer
        )

    @check_type
    def greedy_decoder(self, strings: List[str]):
        """
        true case text + segmentation.

        Parameters
        ----------
        strings: List[str]

        Returns
        -------
        result: List[str]
        """

        return self._predict([f'kes benar: {string}' for string in strings])
