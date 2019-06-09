import xgboost as xgb
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
from .._utils._parse_dependency import DependencyGraph
from ..texts.vectorizer import features_crf, features_crf_dependency


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
        Tag a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: tagged string
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        original_string, string = entities_textcleaning(
            string, lowering = self._is_lower
        )
        batch_x = [features_crf(string, index) for index in range(len(string))]
        return [
            (original_string[no], tag)
            for no, tag in enumerate(self._model.predict_single(batch_x))
        ]

    def analyze(self, string):
        """
        Analyze a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: analyzed string
        """
        predicted = self.predict(string)
        return tag_chunk(predicted)

    def print_transitions(self, top_k = 10):
        """
        Print important top-k transitions.

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
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
        Print important top-k features.

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        print('Top-%d positive:' % (top_k))
        state_features(Counter(self._model.state_features_).most_common(top_k))

        print('\nTop-%d negative:' % (top_k))
        state_features(
            Counter(self._model.state_features_).most_common()[-top_k:]
        )


class DEPENDENCY:
    def __init__(self, tag, depend):
        self._tag = tag
        self._depend = depend

    def predict(self, string):
        """
        Tag a string.

        Parameters
        ----------
        string : str

        Returns
        -------
        string: tagged string
        """
        if not isinstance(string, str):
            raise ValueError('input must be a string')
        original_string, string = entities_textcleaning(string)
        if len(string) > 120:
            raise Exception(
                'Dependency parsing only able to accept string less than 120 words'
            )
        batch_x = [features_crf(string, index) for index in range(len(string))]
        tagging = self._tag.predict_single(batch_x)
        batch_x = [
            features_crf_dependency(string, tagging, index)
            for index in range(len(string))
        ]
        depend = [int(i) for i in self._depend.predict_single(batch_x)]
        for i in range(len(depend)):
            if depend[i] == 0 and tagging[i] != 'root':
                tagging[i] = 'UNK'
            elif depend[i] != 0 and tagging[i] == 'root':
                tagging[i] = 'UNK'
            elif depend[i] > len(tagging):
                depend[i] = len(tagging)
        tagging = [(original_string[i], tagging[i]) for i in range(len(depend))]
        indexing = [(original_string[i], depend[i]) for i in range(len(depend))]
        result = []
        for i in range(len(tagging)):
            result.append(
                '%d\t%s\t_\t_\t_\t_\t%d\t%s\t_\t_'
                % (i + 1, tagging[i][0], int(indexing[i][1]), tagging[i][1])
            )
        d = DependencyGraph('\n'.join(result), top_relation_label = 'root')
        return d, tagging, indexing

    def print_features(self, top_k = 10):
        """
        Print important top-k features for tagging dependency.

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        print('Top-%d tagging positive:' % (top_k))
        state_features(Counter(self._tag.state_features_).most_common(top_k))

        print('\nTop-%d tagging negative:' % (top_k))
        state_features(
            Counter(self._tag.state_features_).most_common()[-top_k:]
        )

    def print_transitions_tag(self, top_k = 10):
        """
        Print important top-k transitions for tagging dependency.

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        print('Top-%d likely tagging transitions:' % (top_k))
        transitions(Counter(self._tag.transition_features_).most_common(top_k))

        print('\nTop-%d unlikely tagging transitions:' % (top_k))
        transitions(
            Counter(self._tag.transition_features_).most_common()[-top_k:]
        )

    def print_transitions_index(self, top_k = 10):
        """
        Print important top-k transitions for indexing dependency.

        Parameters
        ----------
        top_k : int
        """
        if not isinstance(top_k, int):
            raise ValueError('input must be an integer')
        print('Top-%d likely indexing transitions:' % (top_k))
        transitions(
            Counter(self._depend.transition_features_).most_common(top_k)
        )

        print('\nTop-%d unlikely indexing transitions:' % (top_k))
        transitions(
            Counter(self._depend.transition_features_).most_common()[-top_k:]
        )


class XGB:
    def __init__(self, xgb, label, vectorize, cleaning = simple_textcleaning):
        self._xgb = xgb
        self._label = label
        self._vectorize = vectorize
        self._cleaning = cleaning


class BAYES:
    def __init__(
        self, multinomial, label, vectorize, cleaning = simple_textcleaning
    ):
        self._multinomial = multinomial
        self._label = label
        self._vectorize = vectorize
        self._cleaning = cleaning


class BINARY_XGB(XGB):
    def __init__(self, xgb, label, vectorize, cleaning = simple_textcleaning):
        XGB.__init__(self, xgb, label, vectorize, cleaning)

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
        result = self._xgb.predict(
            xgb.DMatrix(vectors), ntree_limit = self._xgb.best_ntree_limit
        )
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
        results = self._xgb.predict(
            xgb.DMatrix(vectors), ntree_limit = self._xgb.best_ntree_limit
        )
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
            return [label[i] for i in np.argmax(results, axis = 1)]


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


class MULTICLASS_XGB(XGB):
    def __init__(self, xgb, label, vectorize, cleaning = simple_textcleaning):
        XGB.__init__(self, xgb, label, vectorize, cleaning)

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
        result = self._xgb.predict(
            xgb.DMatrix(vectors), ntree_limit = self._xgb.best_ntree_limit
        )[0]
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
        results = self._xgb.predict(
            xgb.DMatrix(vectors), ntree_limit = self._xgb.best_ntree_limit
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
        stacked = self._stack([classification_textcleaning(string, True)])
        result = {} if get_proba else []
        for no, label in enumerate(self._class_names):
            if get_proba:
                result[label] = self._models[no].predict_proba(stacked)[0, 1]
            else:
                prob = self._models[no].predict(stacked)[0]
                if prob:
                    result.append(label)
        return result

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
        stacked = self._stack(
            [classification_textcleaning(i, True) for i in strings]
        )
        result = []
        for no in range(len(self._class_names)):
            if get_proba:
                probs = self._models[no].predict_proba(stacked)[:, 1]
            else:
                probs = self._models[no].predict(stacked)
            result.append(probs)
        result = np.array(result).T
        dicts = []
        for row in result:
            nested = {} if get_proba else []
            for no, label in enumerate(self._class_names):
                if get_proba:
                    nested[label] = row[no]
                else:
                    if row[no]:
                        nested.append(label)
            dicts.append(nested)
        return dicts


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
