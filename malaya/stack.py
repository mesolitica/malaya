from scipy.stats.mstats import gmean, hmean, hdmedian
import numpy as np


def _most_common(l):
    return max(set(l), key = l.count)


def voting_stack(models, text):
    """
    Stacking for POS and Entities Recognition models.

    Parameters
    ----------
    models: list
        list of models
    text: str
        string to predict

    Returns
    -------
    result: list
    """
    assert isinstance(models, list), 'models must be a list'
    assert isinstance(text, str), 'text must be a string'
    results, texts, votes = [], [], []
    for i in range(len(models)):
        assert 'predict' in dir(models[i]), 'all models must able to predict'
        predicted = np.array(models[i].predict(text))
        results.append(predicted[:, 1:2])
        texts.append(predicted[:, 0])
    concatenated = np.concatenate(results, axis = 1)
    for row in concatenated:
        votes.append(_most_common(row.tolist()))
    return list(map(lambda X: (X[0], X[1]), list(zip(texts[-1], votes))))


def predict_stack(models, text, mode = 'gmean'):
    """
    Stacking for predictive models.

    Parameters
    ----------
    models: list
        list of models
    text: str
        string to predict
    mode : str, optional (default='gmean')
        Model architecture supported. Allowed values:

        * ``'gmean'`` - geometrical mean
        * ``'hmean'`` - harmonic mean
        * ``'mean'`` - mean
        * ``'min'`` - min
        * ``'max'`` - max
        * ``'median'`` - Harrell-Davis median


    Returns
    -------
    result: dict
    """
    assert isinstance(models, list), 'models must be a list'
    assert isinstance(text, str), 'text must be a string'
    assert isinstance(mode, str), 'mode must be a string'
    if mode.lower() == 'gmean':
        mode = gmean
    elif mode.lower() == 'hmean':
        mode = hmean
    elif mode.lower() == 'mean':
        mode = np.mean
    elif mode.lower() == 'min':
        mode = np.amin
    elif mode.lower() == 'max':
        mode = np.amax
    elif mode.lower() == 'median':
        mode = hdmedian
    else:
        raise Exception(
            "mode not supported, only support ['gmean','hmean','mean','min','max','median']"
        )
    labels, results = [], []
    for i in range(len(models)):
        nested_results = []
        assert 'predict' in dir(models[i]), 'all models must able to predict'
        result = (
            models[i].predict(text)
            if models[i].predict.__defaults__ is None
            else models[i].predict(text, get_proba = True)
        )
        for key, item in result.items():
            if 'attention' in key:
                continue
            if key not in labels:
                labels.append(key)
            nested_results.append(item)
        results.append(nested_results)
    results = mode(np.array(results), axis = 0)
    return {label: results[no] for no, label in enumerate(labels)}
