from scipy.stats.mstats import gmean, hmean, hdmedian
import numpy as np
from herpetologist import check_type
from typing import List, Callable


def _most_common(l):
    return max(set(l), key=l.count)


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
        if 'predict' not in dir(models[i]):
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
    concatenated = np.concatenate(results, axis=1)
    for row in concatenated:
        votes.append(_most_common(row.tolist()))
    if is_dependency:
        concatenated = np.concatenate(indices, axis=1)
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


@check_type
def predict_stack(
    models, strings: List[str], aggregate: Callable = gmean, **kwargs
):
    """
    Stacking for predictive models.

    Parameters
    ----------
    models: List[Callable]
        list of models.
    strings: List[str]
    aggregate : Callable, optional (default=scipy.stats.mstats.gmean)
        Aggregate function.

    Returns
    -------
    result: dict
    """
    if not isinstance(models, list):
        raise ValueError('models must be a list')
    mode = aggregate

    for i in range(len(models)):
        if 'predict_proba' not in dir(models[i]):
            raise ValueError('all models must able to `predict_proba`')

    labels, results = None, []
    for i in range(len(models)):
        nested_results = []
        result = models[i].predict_proba(strings, **kwargs)
        for r in result:
            if isinstance(r, dict):
                l = list(r.keys())
                if not labels:
                    labels = l
                else:
                    if l != labels:
                        raise ValueError('domain classification must be same!')
                nested_results.append(list(r.values()))
            else:
                nested_results.append(r)
        results.append(nested_results)
    results = mode(np.array(results), axis=0)
    outputs = []
    if labels:
        for result in results:
            outputs.append(
                {label: result[no] for no, label in enumerate(labels)}
            )
    else:
        outputs = results
    return outputs
