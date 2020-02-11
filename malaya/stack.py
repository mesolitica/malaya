from scipy.stats.mstats import gmean, hmean, hdmedian
import numpy as np
from herpetologist import check_type
from typing import List


def _most_common(l):
    return max(set(l), key = l.count)


@check_type
def voting_stack(models, text: str):
    """
    Stacking for POS, Entities and Dependency models.

    Parameters
    ----------
    models: list
        list of models.
    text: str
        string to predict.

    Returns
    -------
    result: list
    """
    if not isinstance(models, list):
        raise ValueError('models must be a list')
    results, texts, votes, votes_indices, indices = [], [], [], [], []
    is_dependency = False
    for i in range(len(models)):
        if not 'predict' in dir(models[i]):
            raise ValueError('all models must able to predict')
        predicted = models[i].predict(text)
        if isinstance(predicted, tuple):
            is_dependency = True
            d, predicted, indexing = predicted
            indexing = np.array(indexing)
            indices.append(indexing[:, 1:2])

        predicted = np.array(predicted)
        results.append(predicted[:, 1:2])
        texts.append(predicted[:, 0])
    concatenated = np.concatenate(results, axis = 1)
    for row in concatenated:
        votes.append(_most_common(row.tolist()))
    if is_dependency:
        concatenated = np.concatenate(indices, axis = 1)
        for row in concatenated:
            votes_indices.append(_most_common(row.tolist()))
    output = list(map(lambda X: (X[0], X[1]), list(zip(texts[-1], votes))))
    if is_dependency:
        return (
            output,
            list(
                map(lambda X: (X[0], X[1]), list(zip(texts[-1], votes_indices)))
            ),
        )
    else:
        return output


dict_function = {
    'gmean': gmean,
    'hmean': hmean,
    'mean': np.mean,
    'min': np.amin,
    'max': np.amax,
    'median': hdmedian,
}


@check_type
def predict_stack(models, strings, mode: str = 'gmean'):
    """
    Stacking for predictive models.

    Parameters
    ----------
    models: list
        list of models.
    strings: str or list of str
        strings to predict.
    mode : str, optional (default='gmean')
        Model architecture supported. Allowed values:

        * ``'gmean'`` - geometrical mean.
        * ``'hmean'`` - harmonic mean.
        * ``'mean'`` - mean.
        * ``'min'`` - min.
        * ``'max'`` - max.
        * ``'median'`` - Harrell-Davis median.


    Returns
    -------
    result: dict
    """
    if not isinstance(models, list):
        raise ValueError('models must be a list')
    if isinstance(strings, list):
        if not isinstance(strings[0], str):
            raise ValueError('input must be a list of strings or a string')
    else:
        if not isinstance(strings, str):
            raise ValueError('input must be a list of strings or a string')
    if isinstance(strings, str):
        strings = [strings]
    if mode.lower() not in dict_function:
        raise Exception(
            "mode not supported, only support ['gmean','hmean','mean','min','max','median']"
        )
    mode = dict_function[mode.lower()]

    for i in range(len(models)):
        if not 'predict_batch' in dir(models[i]):
            raise ValueError('all models must able to predict_batch')

    labels, results = None, []
    for i in range(len(models)):
        nested_results = []
        result = (
            models[i].predict_batch(strings)
            if models[i].predict.__defaults__ is None
            else models[i].predict_batch(strings, get_proba = True)
        )
        for r in result:
            l = list(r.keys())
            if not labels:
                labels = l
            else:
                if l != labels:
                    raise ValueError('domain classification must be same!')
            nested_results.append(list(r.values()))
        results.append(nested_results)
    results = mode(np.array(results), axis = 0)
    outputs = []
    for result in results:
        outputs.append({label: result[no] for no, label in enumerate(labels)})
    return outputs
