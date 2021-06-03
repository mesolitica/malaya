import itertools


def _pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    sequence = iter(sequence)
    if pad_left:
        sequence = itertools.chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = itertools.chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(
    sequence,
    n: int,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    generate ngrams.

    Parameters
    ----------
    sequence : List[str]
        list of tokenize words.
    n : int
        ngram size

    Returns
    -------
    result: List[Tuple[str, str]]
    """
    sequence = _pad_sequence(
        sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
    )

    history = []
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            return
        history.append(next_item)
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]
