import tensorflow as tf
from malaya.text.rouge import postprocess_summary
from malaya.text.function import summarization_textcleaning
from malaya.model.abstract import Seq2Seq
from herpetologist import check_type
from typing import List

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences


class Summarization(Seq2Seq):
    def __init__(self, input_nodes, output_nodes, sess, tokenizer):

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._tokenizer = tokenizer

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
        batch_x = pad_sequences(batch_x, padding='post')

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
        temperature: float = 0.2,
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
