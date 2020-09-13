import json
from malaya.function import check_file
from malaya.supervised import softmax
from malaya.lexicon import nsfw
from malaya.path import PATH_NSFW, S3_PATH_NSFW

label = ['sex', 'gambling', 'negative']


def lexicon(**kwargs):
    """
    Load Lexicon NSFW model.

    Returns
    -------
    result : malaya.lexicon.nsfw.LEXICON class
    """

    check_file(PATH_NSFW['lexicon'], S3_PATH_NSFW['lexicon'], **kwargs)
    with open(PATH_NSFW['lexicon']['model']) as fopen:
        corpus = json.load(fopen)
    return nsfw.LEXICON(corpus)


def multinomial(**kwargs):
    """
    Load multinomial NSFW model.

    Returns
    -------
    result : malaya.model.ml.BAYES class
    """
    return softmax.multinomial(PATH_NSFW, S3_PATH_NSFW, 'nsfw', label, **kwargs)
