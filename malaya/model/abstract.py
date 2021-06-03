from malaya.function.execute import execute_graph
from malaya.text.function import pad_sentence_batch, translation_textcleaning


class Abstract:
    def _execute(self, inputs, input_labels, output_labels):
        return execute_graph(
            inputs=inputs,
            input_labels=input_labels,
            output_labels=output_labels,
            sess=self._sess,
            input_nodes=self._input_nodes,
            output_nodes=self._output_nodes,
        )


class Seq2Seq(Abstract):
    def greedy_decoder(self, strings, **kwargs):
        raise NotImplementedError

    def beam_decoder(self, strings, **kwargs):
        raise NotImplementedError

    def nucleus_decoder(self, strings, **kwargs):
        raise NotImplementedError


class T2T:
    def __init__(
        self,
        input_nodes,
        output_nodes,
        sess,
        encoder,
        translation_model=False,
    ):

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes
        self._sess = sess
        self._encoder = encoder
        self._translation_model = translation_model

    def _predict(self, strings, beam_search=True):
        if self._translation_model:
            encoded = [
                self._encoder.encode(translation_textcleaning(string)) + [1]
                for string in strings
            ]
        else:
            encoded = self._encoder.encode(strings)
        batch_x = pad_sentence_batch(encoded, 0)[0]

        if beam_search:
            output = 'beam'
        else:
            output = 'greedy'

        r = self._execute(
            inputs=[batch_x],
            input_labels=['Placeholder'],
            output_labels=[output],
        )
        p = r[output].tolist()
        if self._translation_model:
            result = []
            for row in p:
                result.append(
                    self._encoder.decode([i for i in row if i not in [0, 1]])
                )
        else:
            result = self._encoder.decode(p)
        return result

    def _greedy_decoder(self, strings):
        return self._predict(strings, beam_search=False)

    def _beam_decoder(self, strings):
        return self._predict(strings, beam_search=True)


class Classification(Abstract):
    def vectorize(self, strings, **kwargs):
        raise NotImplementedError

    def predict(self, strings, **kwargs):
        raise NotImplementedError

    def predict_proba(self, strings, **kwargs):
        raise NotImplementedError

    def predict_words(self, string, **kwargs):
        raise NotImplementedError


class Tagging(Abstract):
    def vectorize(self, string, **kwargs):
        raise NotImplementedError

    def predict(self, string, **kwargs):
        raise NotImplementedError

    def analyze(self, string, **kwargs):
        raise NotImplementedError
