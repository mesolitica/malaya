import xgboost as xgb
import numpy as np
from collections import Counter
from scipy.sparse import hstack
from .text_functions import (
    simple_textcleaning,
    classification_textcleaning,
    features_crf,
    entities_textcleaning,
)

from collections import Counter


def transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print('%-6s -> %-7s %0.6f' % (label_from, label_to, weight))


def state_features(state_features):
    for (attr, label), weight in state_features:
        print('%0.6f %-8s %s' % (weight, label, attr))


class CRF:
    def __init__(self, model, is_lower = True):
        self._model = model
        self._is_lower = is_lower

    def predict(self, string):
        """
        Tag a string

        Parameters
        ----------
        string : str

        Returns
        -------
        string: tagged string
        """
        assert isinstance(string, str), 'input must be a string'
        string = string.lower() if self._is_lower else string
        string = entities_textcleaning(string)
        batch_x = [features_crf(string, index) for index in range(len(string))]
        return [
            (string[no], tag)
            for no, tag in enumerate(self._model.predict_single(batch_x))
        ]

    def print_transitions(self, top_k = 10):
        """
        Print important top-k transitions

        Parameters
        ----------
        top_k : int
        """
        assert isinstance(top_k, int), 'input must be an integer'
        print('Top-%d likely transitions:' % (top_k))
        transitions(
            Counter(self._model.transition_features_).most_common(top_k)
        )

        print('\nTop-%d unlikely transitions:' % (top_k))
        transitions(
            Counter(self._model.transition_features_).most_common()[-top_k:]
        )

    def print_features(self, top_k = 10):
        """
        Print important top-k features

        Parameters
        ----------
        top_k : int
        """
        assert isinstance(top_k, int), 'input must be an integer'
        print('Top-%d positive:' % (top_k))
        state_features(Counter(self._model.state_features_).most_common(top_k))

        print('\nTop-%d negative:' % (top_k))
        state_features(
            Counter(self._model.state_features_).most_common()[-top_k:]
        )


class USER_XGB:
    def __init__(self, xgb, label, vectorize, cleaning = simple_textcleaning):
        self.xgb = xgb
        self.label = label
        self.vectorize = vectorize
        self._cleaning = cleaning

    def predict(self, string, get_proba = False):
        """
        Classify a string

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: result
        """
        assert isinstance(string, str), 'input must be a string'
        vectors = self.vectorize.transform([self._cleaning(string)])
        result = self.xgb.predict(
            xgb.DMatrix(vectors), ntree_limit = self.xgb.best_ntree_limit
        )[0]
        if get_proba:
            return {self.label[i]: result[i] for i in range(len(result))}
        else:
            return self.label[np.argmax(result)]

    def predict_batch(self, strings, get_proba = False):
        """
        Classify a list of strings

        Parameters
        ----------
        strings: list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: list of results
        """
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        strings = [self._cleaning(string) for string in strings]
        vectors = self.vectorize.transform(strings)
        results = self.xgb.predict(
            xgb.DMatrix(vectors), ntree_limit = self.xgb.best_ntree_limit
        )
        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {self.label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [self.label[i] for i in np.argmax(results, axis = 1)]


class USER_BAYES:
    def __init__(
        self, multinomial, label, vectorize, cleaning = simple_textcleaning
    ):
        self.multinomial = multinomial
        self.label = label
        self.vectorize = vectorize
        self._cleaning = cleaning

    def predict(self, string, get_proba = False):
        """
        Classify a string

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: result
        """
        assert isinstance(string, str), 'input must be a string'
        vectors = self.vectorize.transform([self._cleaning(string)])
        if get_proba:
            result = self.multinomial.predict_proba(vectors)[0]
            return {self.label[i]: result[i] for i in range(len(result))}
        else:
            return self.label[self.multinomial.predict(vectors)[0]]

    def predict_batch(self, strings, get_proba = False):
        """
        Classify a list of strings

        Parameters
        ----------
        strings: list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: list of results
        """
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        strings = [self._cleaning(string) for string in strings]
        vectors = self.vectorize.transform(strings)
        if get_proba:
            results = self.multinomial.predict_proba(vectors)
            outputs = []
            for result in results:
                outputs.append(
                    {self.label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [
                self.label[result]
                for result in self.multinomial.predict(vectors)
            ]


class TOXIC:
    def __init__(self, models, vectors):
        self._models = models
        self._vectors = vectors
        self._class_names = [
            'toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate',
        ]

    def _stack(self, strings):
        char_features = self._vectors['char'].transform(strings)
        word_features = self._vectors['word'].transform(strings)
        return hstack([char_features, word_features])

    def predict(self, string, get_proba = False):
        """
        Classify a string

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: result
        """
        assert isinstance(string, str), 'input must be a string'
        stacked = self._stack([classification_textcleaning(string, True)])
        result = {}
        for no, label in enumerate(self._class_names):
            if get_proba:
                result[label] = self._models[no].predict_proba(stacked)[0, 1]
            else:
                result[label] = self._models[no].predict(stacked)[0]
        return result

    def predict_batch(self, strings, get_proba = False):
        """
        Classify a list of strings

        Parameters
        ----------
        strings: list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: list of results
        """
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        stacked = self._stack(
            [classification_textcleaning(i, True) for i in strings]
        )
        result = {}
        for no, label in enumerate(self._class_names):
            if get_proba:
                result[label] = (
                    self._models[no].predict_proba(stacked)[:, 1].tolist()
                )
            else:
                result[label] = self._models[no].predict(stacked).tolist()
        return result


class LANGUAGE_DETECTION:
    def __init__(self, model, label, vectorizer, mode = 'sklearn'):
        self._model = model
        self._label = label
        self._vectorizer = vectorizer
        self._mode = mode

    def predict(self, string, get_proba = False):
        """
        Classify a string

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: result
        """
        assert isinstance(string, str), 'input must be a string'
        string = string.lower()
        char_features = self._vectorizer['bow'].transform([string])
        tfidf_features = self._vectorizer['tfidf'].transform([string])
        vectors = hstack([char_features, tfidf_features])
        if self._mode == 'xgb':
            result = self._model.predict(
                xgb.DMatrix(vectors), ntree_limit = self._model.best_ntree_limit
            )[0]
            if get_proba:
                return {self._label[i]: result[i] for i in range(len(result))}
            else:
                return self._label[np.argmax(result)]
        else:
            if get_proba:
                result = self._model.predict_proba(vectors)[0]
                return {self._label[i]: result[i] for i in range(len(result))}
            else:
                return self._label[self._model.predict(vectors)[0]]

    def predict_batch(self, strings, get_proba = False):
        """
        Classify a list of strings

        Parameters
        ----------
        strings: list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: list of results
        """
        assert isinstance(strings, list) and isinstance(
            strings[0], str
        ), 'input must be list of strings'
        strings = [string.lower() for string in strings]
        char_features = self._vectorizer['bow'].transform(strings)
        tfidf_features = self._vectorizer['tfidf'].transform(strings)
        vectors = hstack([char_features, tfidf_features])

        if self._mode == 'xgb':
            results = self._model.predict(
                xgb.DMatrix(vectors), ntree_limit = self._model.best_ntree_limit
            )
            if get_proba:
                outputs = []
                for result in results:
                    outputs.append(
                        {self._label[i]: result[i] for i in range(len(result))}
                    )
                return outputs
            else:
                return [self._label[i] for i in np.argmax(results, axis = 1)]
        else:
            if get_proba:
                results = self._model.predict_proba(vectors)
                outputs = []
                for result in results:
                    outputs.append(
                        {self._label[i]: result[i] for i in range(len(result))}
                    )
                return outputs
            else:
                return [
                    self._label[result]
                    for result in self._model.predict(vectors)
                ]
