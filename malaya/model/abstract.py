class Seq2Seq:
    def greedy_decoder(self, strings, **kwargs):
        raise NotImplementedError

    def beam_decoder(self, strings, **kwargs):
        raise NotImplementedError

    def nucleus_decoder(self, strings, **kwargs):
        raise NotImplementedError


class Classification:
    def vectorize(self, strings, **kwargs):
        raise NotImplementedError

    def predict(self, strings, **kwargs):
        raise NotImplementedError

    def predict_proba(self, strings, **kwargs):
        raise NotImplementedError

    def predict_words(self, string, **kwargs):
        raise NotImplementedError


class Tagging:
    def vectorize(self, string, **kwargs):
        raise NotImplementedError

    def predict(self, string, **kwargs):
        raise NotImplementedError

    def analyze(self, string, **kwargs):
        raise NotImplementedError
