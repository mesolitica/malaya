import tensorflow as tf
from malaya.text.function import (
    transformer_textcleaning,
    summarization_textcleaning,
    split_into_sentences,
    upperfirst,
)
from malaya.text.rouge import postprocess_summary
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
        summaries = self._predict([f'{mode}: {summarization_textcleaning(string)}' for string in strings])
        if postprocess and mode != 'tajuk':
            summaries = [postprocess_summary(strings[no], summary, **kwargs) for no, summary in enumerate(summaries)]
        return summaries

    @check_type
    def greedy_decoder(
        self,
        strings: List[str],
        mode: str = 'ringkasan',
        postprocess: bool = False,
        **kwargs,
    ):
        """
        Summarize strings. Decoder is greedy decoder with beam width size 1, alpha 0.5 .

        Parameters
        ----------
        strings: List[str]
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * ``'tajuk'`` - title summarization for long sentence, eg, news title.
        postprocess: bool, optional (default=False)
            If True, will filter sentence generated using ROUGE score and removed international news publisher.

        Returns
        -------
        result: List[str]
        """
        mode = mode.lower()
        if mode not in ['ringkasan', 'tajuk']:
            raise ValueError('mode only supports [`ringkasan`, `tajuk`]')

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

    def _paraphrase(self, strings):

        paraphrases = self._predict([f'parafrasa: {summarization_textcleaning(string)}' for string in strings])
        return [upperfirst(paraphrase) for paraphrase in paraphrases]

    @check_type
    def greedy_decoder(self, strings: List[str], split_fullstop: bool = True):
        """
        paraphrase strings. Decoder is greedy decoder with beam width size 1, alpha 0.5 .

        Parameters
        ----------
        strings: List[str]
        split_fullstop: bool, (default=True)
            if True, will generate paraphrase for each strings splitted by fullstop.

        Returns
        -------
        result: List[str]
        """
        results = []

        for string in strings:

            if split_fullstop:

                splitted_fullstop = split_into_sentences(string)

                output = []
                for splitted in splitted_fullstop:
                    if len(splitted.split()) < 4:
                        output.append(splitted)
                    else:
                        output.append(self._paraphrase(splitted))
                r = ' '.join(output)

            else:
                r = self._paraphrase(string)
            results.append(r)

        return results
