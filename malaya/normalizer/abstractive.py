from malaya.supervised import huggingface as load_huggingface
from malaya.function import describe_availability
from typing import Callable
import logging

logger = logging.getLogger(__name__)

_huggingface_availability = {
    'mesolitica/finetune-noisy-translation-t5-tiny-bahasa-cased-v2': {
        'Size (MB)': 139,
        'BLEU': 60.0009672168891,
        'SacreBLEU Verbose': '77.9/63.9/54.6/47.7 (BP = 1.000 ratio = 1.036 hyp_len = 110970 ref_len = 107150)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-noisy-translation-t5-small-bahasa-cased-v4': {
        'Size (MB)': 242,
        'BLEU': 64.06258219941243,
        'SacreBLEU Verbose': '80.1/67.7/59.1/52.5 (BP = 1.000 ratio = 1.042 hyp_len = 111635 ref_len = 107150)',
        'Suggested length': 256,
    },
    'mesolitica/finetune-noisy-translation-t5-base-bahasa-cased-v2': {
        'Size (MB)': 892,
        'BLEU': 64.583819005204,
        'SacreBLEU Verbose': '80.2/68.1/59.8/53.2 (BP = 1.000 ratio = 1.048 hyp_len = 112260 ref_len = 107150)',
        'Suggested length': 256,
    },
}


def available_huggingface():
    """
    List available huggingface models.
    """

    logger.info('tested on noisy twitter google translation, https://huggingface.co/datasets/mesolitica/augmentation-test-set')
    return describe_availability(_huggingface_availability)


def huggingface(
    model: str = 'mesolitica/finetune-noisy-translation-t5-small-bahasa-cased-v4',
    force_check: bool = True,
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
    model: str, optional (default='mesolitica/finetune-noisy-translation-t5-small-bahasa-cased-v4')
        Check available models at `malaya.normalizer.abstractive.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.
    use_rules_normalizer: bool, optional(default=True)
    kenlm_model: str, optional (default='bahasa-wiki-news')
        the model trained on `malaya.language_model.kenlm(model = 'bahasa-wiki-news')`,
        but you can use any kenlm model from `malaya.language_model.available_kenlm`.
        Also you can pass as None to skip spelling correction but still apply rules normalizer.
        This parameter will be ignored if `use_rules_normalizer=False`.
    stem_model: str, optional (default='noisy')
        the model trained on `malaya.stem.deep_model(model = 'noisy'),
        but you can use any stemmer model from `malaya.stem.available_model`.
        Also you can pass as None to skip stemming but still apply rules normalizer.
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

        from malaya.language_model import kenlm
        from malaya.stem import deep_model
        from malaya.spelling_correction.probability import load as load_spelling
        from malaya.normalizer.rules import load as load_normalizer

        if isinstance(kenlm_model, str):
            lm = kenlm(model=kenlm_model)
        else:
            lm = None
        if isinstance(stem_model, str):
            stemmer = deep_model(model=stem_model)
        else:
            stemmer = None
        corrector = load_spelling(
            language_model=lm,
            replace_augmentation=replace_augmentation,
            stemmer=stemmer,
            maxlen=maxlen_speller,
            minlen=minlen_speller,
        )
        normalizer = load_normalizer(corrector, stemmer, date=False)
    else:
        normalizer = None

    return load_huggingface.load_normalizer(
        model=model,
        initial_text='terjemah pasar Melayu ke Melayu: ',
        normalizer=normalizer,
        segmenter=segmenter,
        text_scorer=text_scorer,
        **kwargs,
    )
