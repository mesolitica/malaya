from malaya.function import check_file
from malaya.text.function import classification_textcleaning_stemmer
from malaya.stem import naive
from malaya.text.bpe import (
    WordPieceTokenizer,
    YTTMEncoder,
)
from malaya.model.ml import BinaryBayes, MulticlassBayes, MultilabelBayes
from functools import partial
import os
import pickle


def multinomial(path, s3_path, module, label, sigmoid=False, **kwargs):
    path = check_file(path['multinomial'], s3_path['multinomial'], **kwargs)
    with open(path['model'], 'rb') as fopen:
        multinomial = pickle.load(fopen)
    with open(path['vector'], 'rb') as fopen:
        vectorize = pickle.load(fopen)

    bpe = YTTMEncoder(vocab_file=path['bpe'])

    stemmer = naive()
    cleaning = partial(classification_textcleaning_stemmer, stemmer=stemmer)

    if sigmoid:
        selected_model = MultilabelBayes
    else:
        if len(label) > 2:
            selected_model = MulticlassBayes
        else:
            selected_model = BinaryBayes

    return selected_model(
        multinomial=multinomial,
        label=label,
        vectorize=vectorize,
        bpe=bpe,
        cleaning=cleaning,
    )
