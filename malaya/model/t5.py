import tensorflow as tf
from herpetologist import check_type
from malaya.text.function import transformer_textcleaning


class Summarization:
    def __init__(self, model):
        self._model = model

    def _summarize(self, string, mode):
        string = f'{mode}: string'
        return string

    @check_type
    def summarize(self, corpus, mode: str = 'ringkasan'):
        """
        Summarize list of strings / corpus.

        Parameters
        ----------
        corpus: str, list
        mode: str
            mode for summarization. Allowed values:

            * ``'ringkasan'`` - summarization for long sentence, eg, news summarization.
            * `'tajuk'`` - title summarization for long sentence, eg, news title.
            * `'pendek'`` - summarization for each sentences. This will automatically split sentences by EOS.

        Returns
        -------
        string: summarized string
        """
        mode = mode.lower()
        if mode not in ['ringkasan', 'tajuk', 'pendek']:
            raise ValueError(
                'mode only supports `ringkasan`, `tajuk`, and `pendek`'
            )
        if not isinstance(corpus, list) and not isinstance(corpus, str):
            raise ValueError('corpus must be a list')
        if isinstance(corpus, list):
            if not isinstance(corpus[0], str):
                raise ValueError('corpus must be list of strings')

        if isinstance(corpus, str):
            corpus = split_into_sentences(corpus)
        else:
            corpus = '. '.join(corpus)
            corpus = split_into_sentences(corpus)

        splitted_fullstop = [transformer_textcleaning(i) for i in corpus]

        if mode == 'pendek':
            results = []
            for splitted in splitted_fullstop:
                if len(splitted.split()) < 5:
                    results.append(splitted)
                else:
                    results.append(self._summarize(splitted, mode))
            results = '. '.join(results)

        else:
            joined = '. '.join(splitted_fullstop)
            results = self._summarize(joined, mode)

        return results
