import tensorflow as tf
from malaya.torch_model.huggingface import (
    Generator,
    Prefix,
    Paraphrase,
    Summarization,
    Similarity,
    ZeroShotClassification,
    ZeroShotNER,
    ExtractiveQA,
    AbstractiveQA,
    Transformer,
    IsiPentingGenerator,
    Tatabahasa,
    Normalizer,
    Keyword,
    Dependency,
    TexttoKG,
    KGtoText,
    Translation,
)
from malaya.torch_model.llm import LLM
from transformers import TFAutoModel, AutoTokenizer
from transformers import AutoTokenizer
from malaya_boilerplate.utils import check_tf2


@check_tf2
def load_automodel(model, model_class, huggingface_class=None, **kwargs):

    tokenizer = AutoTokenizer.from_pretrained(model)
    if huggingface_class is None:
        huggingface_class = TFAutoModel
    model = huggingface_class.from_pretrained(model)
    return model_class(model=model, tokenizer=tokenizer, **kwargs)


def load_generator(model, initial_text, **kwargs):
    return Generator(model, initial_text, **kwargs)


def load_prefix(model, **kwargs):
    return Prefix(model, **kwargs)


def load_paraphrase(model, initial_text, **kwargs):
    return Paraphrase(model, initial_text, **kwargs)


def load_summarization(model, initial_text, **kwargs):
    return Summarization(model, initial_text, **kwargs)


def load_similarity(model, **kwargs):
    return Similarity(model=model, **kwargs)


def load_zeroshot_classification(model, **kwargs):
    return ZeroShotClassification(model=model, **kwargs)


def load_zeroshot_ner(model, **kwargs):
    return ZeroShotNER(model=model, **kwargs)


def load_extractive_qa(model, **kwargs):
    return ExtractiveQA(model=model, flan_mode='flan' in model, **kwargs)


def load_abstractive_qa(model, **kwargs):
    return AbstractiveQA(model=model, **kwargs)


def load_transformer(model, **kwargs):
    return Transformer(model=model, **kwargs)


def load_isi_penting(model, **kwargs):
    return IsiPentingGenerator(model, **kwargs)


def load_tatabahasa(model, initial_text, **kwargs):
    return Tatabahasa(model, initial_text, **kwargs)


def load_normalizer(
    model,
    initial_text,
    normalizer,
    segmenter=None,
    text_scorer=None,
    **kwargs,
):
    return Normalizer(
        model,
        initial_text,
        normalizer,
        segmenter=segmenter,
        text_scorer=text_scorer,
        **kwargs,
    )


def load_keyword(model, **kwargs):
    return Keyword(model, **kwargs)


def load_dependency(model, **kwargs):
    return Dependency(model, **kwargs)


def load_ttkg(model, **kwargs):
    return TexttoKG(model=model, **kwargs)


def load_kgtt(model, **kwargs):
    return KGtoText(model=model, **kwargs)


def load_translation(model, from_lang, to_lang, **kwargs):
    return Translation(model=model, from_lang=from_lang, to_lang=to_lang, **kwargs)


def load_llm(base_model, model, lora, **kwargs):
    return LLM(base_model=base_model, model=model, lora=lora, **kwargs)
