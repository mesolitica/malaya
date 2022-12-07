import tensorflow as tf
import json
import re
from functools import partial
from malaya.path import PATH_NGRAM, S3_PATH_NGRAM
from malaya.function import check_file
from malaya.supervised import t5 as t5_load
from malaya.model.t5 import Spell as T5_Spell
from malaya.text.bpe import SentencePieceTokenizer
from malaya.text.function import case_of, check_ratio_upper_lower
from malaya.dictionary import is_english, is_malay
from malaya.text.tatabahasa import stopword_tatabahasa
from malaya.text.rules import rules_normalizer
from malaya.spelling_correction.probability import Spell
from malaya.function import describe_availability
from typing import List
from herpetologist import check_type
import numpy as np
import logging

logger = logging.getLogger(__name__)

_transformer_availability = {
    'small-t5': {
        'Size (MB)': 355.6,
        'Quantized Size (MB)': 195,
        'WER': 0.0156248,
        'Suggested length': 256,
    },
    'tiny-t5': {
        'Size (MB)': 208,
        'Quantized Size (MB)': 103,
        'WER': 0.023712,
        'Suggested length': 256,
    },
    'super-tiny-t5': {
        'Size (MB)': 81.8,
        'Quantized Size (MB)': 27.1,
        'WER': 0.038001,
        'Suggested length': 256,
    },
}


def tokens_to_masked_ids(tokens, mask_ind, tokenizer):
    masked_tokens = tokens[:]
    masked_tokens[mask_ind] = '[MASK]'
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    masked_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    return masked_ids


def generate_ids(mask, tokenizer):
    tokens = tokenizer.tokenize(mask)
    input_ids = [
        tokens_to_masked_ids(tokens, i, tokenizer) for i in range(len(tokens))
    ]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, input_ids, tokens_ids


class Transformer(Spell):
    def __init__(self, model, corpus, sp_tokenizer, stemmer, **kwargs):
        Spell.__init__(self, sp_tokenizer, corpus, stemmer, **kwargs)
        self._model = model
        self._padding = tf.keras.preprocessing.sequence.pad_sequences

    def _correct(self, word, string, index, batch_size=20):
        possible_states = self.edit_candidates(word)
        replaced_masks = []
        for state in possible_states:
            mask = string[:]
            mask[index] = state
            replaced_masks.append(' '.join(mask))
        ids = [
            generate_ids(mask, self._model._tokenizer)
            for mask in replaced_masks
        ]
        tokens, input_ids, tokens_ids = list(zip(*ids))

        indices, ids = [], []
        for i in range(len(input_ids)):
            indices.extend([i] * len(input_ids[i]))
            ids.extend(input_ids[i])

        masked_padded = self._padding(ids, padding='post')
        input_masks = masked_padded.astype('bool').astype('int')
        preds = []
        for i in range(0, len(masked_padded), batch_size):
            index = min(i + batch_size, len(masked_padded))
            batch = masked_padded[i:index]
            batch_mask = input_masks[i:index]
            preds.append(self._model._log_vectorize(batch, batch_mask))

        preds = np.concatenate(preds, axis=0)
        indices = np.array(indices)
        scores = []
        for i in range(len(tokens)):
            filter_preds = preds[indices == i]
            total = np.sum(
                [filter_preds[k, k + 1, x] for k, x in enumerate(tokens_ids[i])]
            )
            scores.append(total)

        prob_scores = np.array(scores) / np.sum(scores)
        probs = list(zip(possible_states, prob_scores))
        probs.sort(key=lambda x: x[1])
        return probs[0][0]

    @check_type
    def correct(
        self,
        word: str,
        string: List[str],
        index: int = -1,
        lookback: int = 5,
        lookforward: int = 5,
        batch_size: int = 20,
        **kwargs,
    ):
        """
        Correct a word within a text, returning the corrected word.

        Parameters
        ----------
        word: str
        string: List[str]
            Tokenized string, `word` must a word inside `string`.
        index: int, optional (default=-1)
            index of word in the string, if -1, will try to use `string.index(word)`.
        lookback: int, optional (default=5)
            N words on the left hand side.
            if put -1, will take all words on the left hand side.
            longer left hand side will take longer to compute.
        lookforward: int, optional (default=5)
            N words on the right hand side.
            if put -1, will take all words on the right hand side.
            longer right hand side will take longer to compute.
        batch_size: int, optional (default=20)
            batch size to insert into model.

        Returns
        -------
        result: str
        """

        if batch_size < 1:
            raise ValueError('batch_size must be bigger than 0')
        if index < 0:
            index = string.index(word)
        else:
            if word.lower() not in string[index].lower():
                raise ValueError('word is not a subset or equal to index of the splitted string')

        if is_english(word):
            return word
        if is_malay(word):
            return word
        if word in stopword_tatabahasa:
            return word

        if word in rules_normalizer:
            word = rules_normalizer[word]
        else:
            if lookback == -1:
                lookback = index
            elif lookback > index:
                lookback = index

            if lookforward == -1:
                lookforward = 0

            left_hand = string[index - lookback: index]
            right_hand = string[index + 1: index + 1 + lookforward]
            string = left_hand + [word] + right_hand
            index = len(left_hand)

            logger.debug(f'left hand side: {left_hand}, right hand side: {right_hand}, index: {index}, word: {word}')

            word = self._correct(word, string, index, batch_size=batch_size)
        return word

    @check_type
    def correct_text(
        self,
        text: str,
        lookback: int = 5,
        lookforward: int = 5,
        batch_size: int = 20
    ):
        """
        Correct all the words within a text, returning the corrected text.

        Parameters
        ----------
        text: str
        lookback: int, optional (default=5)
            N words on the left hand side.
            if put -1, will take all words on the left hand side.
            longer left hand side will take longer to compute.
        lookforward: int, optional (default=5)
            N words on the right hand side.
            if put -1, will take all words on the right hand side.
            longer right hand side will take longer to compute.
        batch_size: int, optional(default=20)
            batch size to insert into model.

        Returns
        -------
        result: str
        """

        string = re.sub(r'[ ]+', ' ', text).strip()
        splitted = string.split()
        for no, word in enumerate(splitted):
            if not word.isupper() and check_ratio_upper_lower(word) < 0.5:
                p = partial(
                    self.correct_match,
                    string=splitted,
                    index=no,
                    lookback=lookback,
                    lookforward=lookforward,
                    batch_size=batch_size
                )
                word = re.sub('[a-zA-Z]+', p, word)
            splitted[no] = word

        return ' '.join(splitted)

    @check_type
    def correct_word(
        self,
        word: str,
        string: List[str],
        index: int = -1,
        lookback: int = 5,
        lookforward: int = 5,
        batch_size: int = 20,
    ):
        """
        Spell-correct word, and preserve proper upper, lower and title case.

        Parameters
        ----------
        word: str
        string: List[str]
            Tokenized string, `word` must a word inside `string`.
        index: int, optional(default=-1)
            index of word in the string, if -1, will try to use `string.index(word)`.
        lookback: int, optional (default=5)
            N words on the left hand side.
            if put -1, will take all words on the left hand side.
            longer left hand side will take longer to compute.
        lookforward: int, optional (default=5)
            N words on the right hand side.
            if put -1, will take all words on the right hand side.
            longer right hand side will take longer to compute.
        batch_size: int, optional(default=20)
            batch size to insert into model.

        Returns
        -------
        result: str
        """

        return case_of(word)(self.correct(
            word.lower(),
            string=string,
            index=index,
            lookback=lookback,
            lookforward=lookforward,
            batch_size=batch_size))

    def correct_match(
        self,
        match,
        string: List[str],
        index: int = -1,
        lookback: int = 5,
        lookforward: int = 5,
        batch_size: int = 20,
    ):
        """
        Spell-correct word in re.match, and preserve proper upper, lower, title case.
        """

        word = match.group()
        if len(word) < 2:
            return word
        return case_of(word)(self.correct(
            word.lower(),
            string=string,
            index=index,
            lookback=lookback,
            lookforward=lookforward,
            batch_size=batch_size))


def available_transformer():
    """
    List available transformer models.
    """

    logger.info('tested on 10k generated dataset at https://github.com/huseinzol05/malaya/tree/master/session/spelling-correction/t5')

    return describe_availability(_transformer_availability)


@check_type
def transformer(model: str = 'small-t5', quantized: bool = False, **kwargs):
    """
    Load a Transformer Spell Corrector.

    Parameters
    ----------
    model : str, optional (default='small-t5')
        Model architecture supported. Allowed values:

        * ``'small-t5'`` - T5 SMALL parameters.
        * ``'tiny-t5'`` - T5 TINY parameters.
        * ``'super-tiny-t5'`` - T5 SUPER TINY parameters.

    quantized : bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: malaya.model.t5.Spell class
    """
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.spell.available_transformer()`.'
        )
    return t5_load.load(
        module='spelling-correction',
        model=model,
        model_class=T5_Spell,
        quantized=quantized,
        **kwargs,
    )


@check_type
def encoder(
    model,
    sentence_piece: bool = False,
    stemmer=None,
    **kwargs,
):
    """
    Load a Transformer Encoder Spell Corrector. Right now only supported BERT and ALBERT.

    Parameters
    ----------
    model: Callable
    sentence_piece: bool, optional (default=False)
        if True, reduce possible augmentation states using sentence piece.
    stemmer: Callable, optional (default=None)
        a Callable object, must have `stem_word` method.

    Returns
    -------
    result: malaya.spelling_correction.transformer.Transformer class
    """

    if not hasattr(model, '_log_vectorize'):
        raise ValueError('model must have `_log_vectorize` method')

    if stemmer is not None:
        if not hasattr(stemmer, 'stem_word'):
            raise ValueError('stemmer must have `stem_word` method')

    tokenizer = None
    if sentence_piece:
        path = check_file(
            PATH_NGRAM['sentencepiece'],
            S3_PATH_NGRAM['sentencepiece'],
            **kwargs
        )

        vocab = path['vocab']
        vocab_model = path['model']
        tokenizer = SentencePieceTokenizer(vocab_file=vocab, spm_model_file=vocab_model)

    path = check_file(PATH_NGRAM[1], S3_PATH_NGRAM[1], **kwargs)
    with open(path['model']) as fopen:
        corpus = json.load(fopen)
    return Transformer(model, corpus, tokenizer, stemmer, **kwargs)
