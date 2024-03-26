from typing import List, Tuple

_accepted_pos = [
    'ADJ',
    'ADP',
    'ADV',
    'ADX',
    'CCONJ',
    'DET',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
]
_accepted_entities = [
    'OTHER',
    'law',
    'location',
    'organization',
    'person',
    'quantity',
    'time',
    'event',
    'X',
]


def cluster_words(list_words: List[str], lowercase: bool = False):
    """
    cluster similar words based on structure, eg, ['mahathir mohamad', 'mahathir'] = ['mahathir mohamad'].
    big O = n^2

    Parameters
    ----------
    list_words: List[str]
    lowercase: bool, optional (default=True)
        if True, will group using lowercase but maintain the original form.

    Returns
    -------
    string: List[str]
    """

    dict_words = {}
    for word in list_words:
        found = False
        for key in dict_words.keys():
            if lowercase:
                check = [
                    word.lower() in inside.lower() for inside in dict_words[key]
                ]
            else:
                check = [word in inside for inside in dict_words[key]]
            if word in key or any(check):
                dict_words[key].append(word)
                found = True
            if key in word:
                dict_words[key].append(word)
        if not found:
            dict_words[word] = [word]
    results = []
    for key, words in dict_words.items():
        results.append(max(list(set([key] + words)), key=len))
    return list(set(results))


def cluster_pos(result: List[Tuple[str, str]]):
    """
    cluster similar POS.

    Parameters
    ----------
    result: List[Tuple[str, str]]

    Returns
    -------
    result: Dict[str, List[str]]
    """

    if not all([i[1] in _accepted_pos for i in result]):
        raise ValueError(
            'elements of result must be a subset or equal of supported POS, please run `malaya.pos.describe()` to get supported POS.'
        )

    output = {p: [] for p in _accepted_pos}
    last_label, words = None, []
    for word, label in result:
        if last_label != label and last_label:
            joined = ' '.join(words)
            if joined not in output[last_label]:
                output[last_label].append(joined)
            words = []
            last_label = label
            words.append(word)

        else:
            if not last_label:
                last_label = label
            words.append(word)
    output[last_label].append(' '.join(words))
    return output


def cluster_entities(result: List[Tuple[str, str]]):
    """
    cluster similar Entities.

    Parameters
    ----------
    result: List[Tuple[str, str]]

    Returns
    -------
    result: Dict[str, List[str]]
    """
    if not all([i[1] in _accepted_entities for i in result]):
        raise ValueError(
            'elements of result must be a subset or equal of supported Entities, please run `malaya.entity.describe` to get supported Entities.'
        )

    output = {e: [] for e in _accepted_entities}
    last_label, words = None, []
    for word, label in result:
        if last_label != label and last_label:
            joined = ' '.join(words)
            if joined not in output[last_label]:
                output[last_label].append(joined)
            words = []
            last_label = label
            words.append(word)

        else:
            if not last_label:
                last_label = label
            words.append(word)
    output[last_label].append(' '.join(words))
    return output


def cluster_tagging(result: List[Tuple[str, str]]):
    """
    cluster any tagging results, as long the data passed `[(string, label), (string, label)]`.

    Parameters
    ----------
    result: List[Tuple[str, str]]

    Returns
    -------
    result: Dict[str, List[str]]
    """

    _, labels = list(zip(*result))

    output = {l: [] for l in labels}
    last_label, words = None, []
    for word, label in result:
        if last_label != label and last_label:
            joined = ' '.join(words)
            if joined not in output[last_label]:
                output[last_label].append(joined)
            words = []
            last_label = label
            words.append(word)

        else:
            if not last_label:
                last_label = label
            words.append(word)
    output[last_label].append(' '.join(words))
    return output
