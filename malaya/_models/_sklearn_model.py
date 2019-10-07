import numpy as np
from collections import Counter
from scipy.sparse import hstack
from ..texts._text_functions import (
    simple_textcleaning,
    classification_textcleaning,
    entities_textcleaning,
    language_detection_textcleaning,
    tag_chunk,
)
from .._utils._utils import add_neutral as neutral


class BAYES:
    def __init__(
        self, multinomial, label, vectorize, cleaning = simple_textcleaning
    ):
        self._multinomial = multinomial
        self._label = label
        self._vectorize = vectorize
        self._cleaning = cleaning


class BINARY_BAYES(BAYES):
    def __init__(
        self, multinomial, label, vectorize, cleaning = simple_textcleaning
    ):
        BAYES.__init__(self, multinomial, label, vectorize, cleaning)

    def predict(self, string, get_proba = False, add_neutral = True):
        """
        Classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        string: result
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        if not isinstance(add_neutral, bool):
            raise ValueError('add_neutral must be a boolean')

        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label
        vectors = self._vectorize.transform([self._cleaning(string)])
        result = self._multinomial.predict_proba(vectors)
        if add_neutral:
            result = neutral(result)
        result = result[0]
        if get_proba:
            return {label[i]: result[i] for i in range(len(result))}
        else:
            return label[np.argmax(result)]

    def predict_batch(self, strings, get_proba = False, add_neutral = True):
        """
        Classify a list of strings.

        Parameters
        ----------
        strings: list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        string: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(strings[0], str):
            raise ValueError('input must be list of strings')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        if not isinstance(add_neutral, bool):
            raise ValueError('add_neutral must be a boolean')

        if add_neutral:
            label = self._label + ['neutral']
        else:
            label = self._label

        strings = [self._cleaning(string) for string in strings]
        vectors = self._vectorize.transform(strings)
        results = self._multinomial.predict_proba(vectors)

        if add_neutral:
            results = neutral(results)

        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [label[result] for result in np.argmax(results, axis = 1)]


class MULTICLASS_BAYES(BAYES):
    def __init__(
        self, multinomial, label, vectorize, cleaning = simple_textcleaning
    ):
        BAYES.__init__(self, multinomial, label, vectorize, cleaning)

    def predict(self, string, get_proba = False):
        """
        Classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: result
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        vectors = self._vectorize.transform([self._cleaning(string)])
        result = self._multinomial.predict_proba(vectors)[0]
        if get_proba:
            return {self._label[i]: result[i] for i in range(len(result))}
        else:
            return self._label[np.argmax(result)]

    def predict_batch(self, strings, get_proba = False):
        """
        Classify a list of strings.

        Parameters
        ----------
        strings: list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(strings[0], str):
            raise ValueError('input must be list of strings')
        if not isinstance(get_proba, bool):
            raise ValueError('get_proba must be a boolean')
        strings = [self._cleaning(string) for string in strings]
        vectors = self._vectorize.transform(strings)
        results = self._multinomial.predict_proba(vectors)
        if get_proba:
            outputs = []
            for result in results:
                outputs.append(
                    {self._label[i]: result[i] for i in range(len(result))}
                )
            return outputs
        else:
            return [
                self._label[result] for result in np.argmax(results, axis = 1)
            ]


class MULTILABEL_BAYES:
    def __init__(self, models, vectors, cleaning = simple_textcleaning):
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
        self._cleaning = cleaning

    def predict(self, string, get_proba = False):
        """
        Classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: result
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        vectors = self._vectorize.transform([self._cleaning(string)])
        result = self._multinomial.predict_proba(vectors)[0]
        arounded = np.around(result)
        results = {} if get_proba else []
        for no, label in enumerate(self._class_names):
            if get_proba:
                results[label] = result[no]
            else:
                prob = arounded[no]
                if prob:
                    results.append(label)
        return results

    def predict_batch(self, strings, get_proba = False):
        """
        Classify a list of strings.

        Parameters
        ----------
        strings: list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(strings[0], str):
            raise ValueError('input must be list of strings')

        strings = [self._cleaning(string) for string in strings]
        vectors = self._vectorize.transform(strings)
        result = self._multinomial.predict_proba(vectors)
        arounded = np.around(result)

        results = []
        for i, row in enumerate(result):
            nested_results = {} if get_proba else []
            for no, label in enumerate(self._class_names):
                if get_proba:
                    nested_results[label] = row[no]
                else:
                    prob = arounded[i, no]
                    if prob:
                        nested_results.append(label)
            results.append(nested_results)
        return results


class LANGUAGE_DETECTION:
    def __init__(self, model, label, vectorizer, mode = 'sklearn'):
        self._model = model
        self._label = label
        self._vectorizer = vectorizer
        self._mode = mode

    def predict(self, string, get_proba = False):
        """
        Classify a string.

        Parameters
        ----------
        string : str
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: result
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        string = language_detection_textcleaning(string)
        vectors = self._vectorizer.transform([string])
        if get_proba:
            result = self._model.predict_proba(vectors)[0]
            return {self._label[i]: result[i] for i in range(len(result))}
        else:
            return self._label[self._model.predict(vectors)[0]]

    def predict_batch(self, strings, get_proba = False):
        """
        Classify a list of strings.

        Parameters
        ----------
        strings: list
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: list of results
        """
        if not isinstance(strings, list):
            raise ValueError('input must be a list')
        if not isinstance(strings[0], str):
            raise ValueError('input must be list of strings')
        strings = [
            language_detection_textcleaning(string) for string in strings
        ]
        vectors = self._vectorizer.transform(strings)

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
                self._label[result] for result in self._model.predict(vectors)
            ]
