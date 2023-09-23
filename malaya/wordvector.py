import numpy as np
import json
from malaya.function import check_file
from malaya.path import PATH_WORDVECTOR, S3_PATH_WORDVECTOR
from typing import List
import logging

logger = logging.getLogger(__name__)

available_wordvector = {
    'news': {
        'Size (MB)': 200.2,
        'Vocab size': 195466,
        'lowercase': True,
        'Description': 'pretrained on cleaned Malay news',
        'dimension': 256,
    },
    'wikipedia': {
        'Size (MB)': 781.7,
        'Vocab size': 763350,
        'lowercase': True,
        'Description': 'pretrained on Malay wikipedia',
        'dimension': 256,
    },
    'socialmedia': {
        'Size (MB)': 1300,
        'Vocab size': 1294638,
        'lowercase': True,
        'Description': 'pretrained on cleaned Malay twitter and Malay instagram',
        'dimension': 256,
    },
    'combine': {
        'Size (MB)': 1900,
        'Vocab size': 1903143,
        'lowercase': True,
        'Description': 'pretrained on cleaned Malay news + Malay social media + Malay wikipedia',
        'dimension': 256,
    },
    'socialmedia-v2': {
        'Size (MB)': 1300,
        'Vocab size': 1294638,
        'lowercase': True,
        'Description': 'pretrained on twitter + lowyat + carigold + b.cari.com.my + facebook + IIUM Confession + Common Crawl',
        'dimension': 256,
    }
}


def load(model: str = 'wikipedia', **kwargs):
    """
    Load pretrained word vectors.

    Parameters
    ----------
    model: str, optional (default='wikipedia')
        Check available models at `malaya.wordvector.available_wordvector`.

    Returns
    -------
    vocabulary: indices dictionary for `vector`.
    vector: np.array, 2D.
    """

    model = model.lower()
    if model not in _wordvector_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.wordvector.available_wordvector`.'
        )

    path = check_file(PATH_WORDVECTOR[model], S3_PATH_WORDVECTOR[model], **kwargs)
    with open(path['vocab']) as fopen:
        vocab = json.load(fopen)
    vector = np.load(path['model'])
    return vocab, vector
