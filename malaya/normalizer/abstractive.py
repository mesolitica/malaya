from malaya.supervised import huggingface as load_huggingface
from malaya.language_model import kenlm
from malaya.stem import deep_model
from malaya.spelling_correction.probability import load_spelling
from malaya.function import describe_availability
from typing import Callable
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info('tested on test set IIUM confession normalization dataset, https://huggingface.co/datasets/mesolitica/IIUM-Confession-abstractive-normalization')
    logger.info('tested on test set Twitter normalization dataset, https://huggingface.co/datasets/mesolitica/twitter-abstractive-normalization')

    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/finetune-normalizer-t5-small-standard-bahasa-cased',
    use_rules_normalizer: bool = True,
    kenlm_model: str = 'bahasa-wiki-news',
    stem_model: str = 'noisy',
    segmenter: Callable = None,
    text_scorer: Callable = None,
    replace_augmentation: bool = True,
    minlen_speller: int = 2,
    maxlen_speller: int = 12,
    **kwargs,
):
    """
    Load HuggingFace model to abstractive text normalizer.
    text -> rules based text normalizer -> abstractive.
    To skip rules based text normalizer, set `use_rules_normalizer=False`.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-normalizer-t5-small-standard-bahasa-cased')
        Check available models at `malaya.normalizer.abstractive.available_huggingface()`.
    use_rules_normalizer: bool, optional(default=True)
    kenlm_model: str, optional (default='bahasa-wiki-news')
        the model trained on `malaya.language_model.kenlm(model = 'bahasa-wiki-news')`,
        but you can use any kenlm model from `malaya.language_model.available_kenlm`.
        This parameter will be ignored if `use_rules_normalizer=False`.
    stem_model: str, optional (default='noisy')
        the model trained on `malaya.stem.deep_model(model = 'noisy'),
        but you can use any stemmer model from `malaya.stem.available_model`.
        This parameter will be ignored if `use_rules_normalizer=False`.
    segmenter: Callable, optional (default=None)
        segmenter function to segment text, read more at https://malaya.readthedocs.io/en/stable/load-normalizer.html#Use-segmenter
        during training session, we use `malaya.segmentation.huggingface()`.
        It is save to set as None.
        This parameter will be ignored if `use_rules_normalizer=False`.
    text_scorer: Callable, optional (default=None)
        text scorer to validate upper case.
        during training session, we use `malaya.language_model.kenlm(model = 'bahasa-wiki-news')`.
        This parameter will be ignored if `use_rules_normalizer=False`.
    replace_augmentation: bool, optional (default=True)
        Use replace norvig augmentation method. Enabling this might generate bigger candidates, hence slower.
        This parameter will be ignored if `use_rules_normalizer=False`.
    minlen_speller: int, optional (default=2)
        minimum length of word to check spelling correction.
        This parameter will be ignored if `use_rules_normalizer=False`.
    maxlen_speller: int, optional (default=12)
        max length of word to check spelling correction.
        This parameter will be ignored if `use_rules_normalizer=False`.

    Returns
    -------
    result: malaya.torch_model.huggingface.Normalizer
    """
    if use_rules_normalizer:
        lm = kenlm(model=kenlm_model)
        stemmer = deep_model(model=stem_model)
        corrector = load_spelling(
            language_model=lm,
            replace_augmentation=replace_augmentation,
            stemmer=stemmer,
            maxlen=maxlen_speller,
            minlen=minlen_speller,
        )
    else:
        corrector = None

    return load_huggingface.load_normalizer(
        model=model,
        initial_text='terjemah pasar Melayu ke Melayu: ',
        corrector=corrector,
        segmenter=segmenter,
        text_scorer=text_scorer,
        **kwargs,
    )
