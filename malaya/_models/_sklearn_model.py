import numpy as np
from ..texts._text_functions import (
    simple_textcleaning,
    classification_textcleaning,
    entities_textcleaning,
    language_detection_textcleaning,
    tag_chunk,
)
from .._utils._utils import add_neutral as neutral
from herpetologist import check_type
from typing import List


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

    @check_type
    def predict(
        self, string: str, get_proba: bool = False, add_neutral: bool = True
    ):
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

    @check_type
    def predict_batch(
        self,
        strings: List[str],
        get_proba: bool = False,
        add_neutral: bool = True,
    ):
        """
        Classify a list of strings.

        Parameters
        ----------
        strings: List[str]
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.
        add_neutral: bool, optional (default=True)
            if True, it will add neutral probability.

        Returns
        -------
        string: list of results
        """

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

    @check_type
    def predict(self, string: str, get_proba: bool = False):
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

    @check_type
    def predict_batch(self, strings: List[str], get_proba: bool = False):
        """
        Classify a list of strings.

        Parameters
        ----------
        strings: List[str]
        get_proba: bool, optional (default=False)
            If True, it will return probability of classes.

        Returns
        -------
        string: list of results
        """
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
        self._multinomial = models
        self._vectorize = vectors
        self._class_names = [
            'toxic',
            'severe_toxic',
            'obscene',
            'threat',
            'insult',
            'identity_hate',
        ]
        self._cleaning = cleaning

    @check_type
    def predict(self, string: str, get_proba: bool = False):
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

    @check_type
    def predict_batch(self, strings: List[str], get_proba: bool = False):
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
    def __init__(self, model, lang_labels):
        self._model = model
        self._labels = list(lang_labels.values())

    def _predict(self, strings):
        strings = [
            language_detection_textcleaning(string) for string in strings
        ]
        return self._model.predict(strings)

    @check_type
    def predict(self, string: str, get_proba: bool = False):
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

        result_labels, result_probs = self._predict([string])
        result_labels = result_labels[0]
        result_probs = result_probs[0]
        if get_proba:
            result = {label: 0.0 for label in self._labels}
            for no, label in enumerate(result_labels):
                label = label.replace('__label__', '')
                result[label] = result_probs[no]
            return result
        else:
            return result_labels[0].replace('__label__', '')

    @check_type
    def predict_batch(self, strings: List[str], get_proba: bool = False):
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

        result_labels, result_probs = self._predict(strings)

        if get_proba:
            outputs = []
            for no, labels in enumerate(result_labels):
                result = {label: 0.0 for label in self._labels}
                for no_, label in enumerate(labels):
                    label = label.replace('__label__', '')
                    result[label] = result_probs[no][no_]
                outputs.append(result)
            return outputs
        else:
            return [
                label[0].replace('__label__', '') for label in result_labels
            ]
