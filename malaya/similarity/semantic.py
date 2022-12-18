from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import SentencePieceTokenizer
from malaya.model.bert import SiameseBERT
from malaya.model.xlnet import SiameseXLNET
from malaya.path import MODEL_VOCAB, MODEL_BPE
from malaya.function import describe_availability
from malaya.supervised import huggingface as load_huggingface
from herpetologist import check_type
import logging
import warnings

logger = logging.getLogger(__name__)

label = ['not similar', 'similar']

_transformer_availability = {
    'bert': {
        'Size (MB)': 423.4,
        'Quantized Size (MB)': 111,
        'macro precision': 0.88315,
        'macro recall': 0.88656,
        'macro f1-score': 0.88405,
    },
    'tiny-bert': {
        'Size (MB)': 56.6,
        'Quantized Size (MB)': 15,
        'macro precision': 0.87210,
        'macro recall': 0.87546,
        'macro f1-score': 0.87292,
    },
    'albert': {
        'Size (MB)': 48.3,
        'Quantized Size (MB)': 12.8,
        'macro precision': 0.87164,
        'macro recall': 0.87146,
        'macro f1-score': 0.87155,
    },
    'tiny-albert': {
        'Size (MB)': 21.9,
        'Quantized Size (MB)': 6,
        'macro precision': 0.82234,
        'macro recall': 0.82383,
        'macro f1-score': 0.82295,
    },
    'xlnet': {
        'Size (MB)': 448.7,
        'Quantized Size (MB)': 119,
        'macro precision': 0.80866,
        'macro recall': 0.76775,
        'macro f1-score': 0.77112,
    },
    'alxlnet': {
        'Size (MB)': 49.0,
        'Quantized Size (MB)': 13.9,
        'macro precision': 0.88756,
        'macro recall': 0.88700,
        'macro f1-score': 0.88727,
    },
}

_vectorizer_mapping = {
    'bert': 'import/bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1:0',
    'tiny-bert': 'import/bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1:0',
    'albert': 'import/bert/encoder/transformer/group_0_11/layer_11/inner_group_0/LayerNorm_1/batchnorm/add_1:0',
    'tiny-albert': 'import/bert/encoder/transformer/group_0_3/layer_3/inner_group_0/LayerNorm_1/batchnorm/add_1:0',
    'xlnet': 'import/model/transformer/layer_11/ff/LayerNorm/batchnorm/add_1:0',
    'alxlnet': 'import/model/transformer/layer_shared_11/ff/LayerNorm/batchnorm/add_1:0',
}

_huggingface_availability = {
    'mesolitica/finetune-mnli-t5-super-tiny-standard-bahasa-cased': {
        'Size (MB)': 50.7,
        'macro precision': 0.74562,
        'macro recall': 0.74574,
        'macro f1-score': 0.74501,
    },
    'mesolitica/finetune-mnli-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 139,
        'macro precision': 0.76584,
        'macro recall': 0.76565,
        'macro f1-score': 0.76542,
    },
    'mesolitica/finetune-mnli-t5-small-standard-bahasa-cased': {
        'Size (MB)': 242,
        'macro precision': 0.78067,
        'macro recall': 0.78063,
        'macro f1-score': 0.78010,
    },
    'mesolitica/finetune-mnli-t5-base-standard-bahasa-cased': {
        'Size (MB)': 892,
        'macro precision': 0.78903,
        'macro recall': 0.79064,
        'macro f1-score': 0.78918,
    },
}


def _describe():
    logger.info('tested on matched dev set translated MNLI, https://huggingface.co/datasets/mesolitica/translated-MNLI')


def available_transformer():
    """
    List available transformer similarity models.
    """

    warnings.warn(
        '`malaya.similarity.semantic.available_transformer` is deprecated, use `malaya.similarity.semantic.available_huggingface` instead', DeprecationWarning)
    _describe()
    logger.warning('test set been leaked during training session.')

    return describe_availability(_transformer_availability)


def available_huggingface():
    """
    List available huggingface models.
    """

    _describe()
    return describe_availability(_huggingface_availability)


def _transformer(
    model, bert_model, xlnet_model, quantized=False, siamese=False, **kwargs
):
    model = model.lower()
    if model not in _transformer_availability:
        raise ValueError(
            'model not supported, please check supported models from `malaya.similarity.available_transformer()`.'
        )

    path = check_file(
        file=model,
        module='similarity',
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
        },
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        selected_model = bert_model
        if siamese:
            selected_node = 'import/bert/pooler/dense/BiasAdd:0'

    if model in ['xlnet', 'alxlnet']:
        selected_model = xlnet_model
        if siamese:
            selected_node = 'import/model_1/sequnece_summary/summary/BiasAdd:0'

    if not siamese:
        selected_node = _vectorizer_mapping[model]

    inputs = ['Placeholder', 'Placeholder_1', 'Placeholder_2']
    outputs = ['logits']
    tokenizer = SentencePieceTokenizer(vocab_file=path['vocab'], spm_model_file=path['tokenizer'])
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra={'vectorizer': selected_node}
    )

    return selected_model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        label=label,
    )


@check_type
def transformer(model: str = 'bert', quantized: bool = False, **kwargs):
    """
    Load Transformer similarity model.

    Parameters
    ----------
    model: str, optional (default='bert')
        Check available models at `malaya.similarity.semantic.available_transformer()`.
    quantized: bool, optional (default=False)
        if True, will load 8-bit quantized model.
        Quantized model not necessary faster, totally depends on the machine.

    Returns
    -------
    result: model
        List of model classes:

        * if `bert` in model, will return `malaya.model.bert.SiameseBERT`.
        * if `xlnet` in model, will return `malaya.model.xlnet.SiameseXLNET`.
    """

    warnings.warn(
        '`malaya.similarity.semantic.transformer` is deprecated, use `malaya.similarity.semantic.huggingface` instead', DeprecationWarning)

    return _transformer(
        model=model,
        bert_model=SiameseBERT,
        xlnet_model=SiameseXLNET,
        quantized=quantized,
        siamese=True,
        **kwargs,
    )


def huggingface(
    model: str = 'mesolitica/finetune-mnli-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to calculate semantic similarity between 2 sentences.

    Parameters
    ----------
    model: str, optional (default='mesolitica/finetune-mnli-t5-small-standard-bahasa-cased')
        Check available models at `malaya.similarity.semantic.available_huggingface()`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Similarity
    """

    if model not in _huggingface_availability and force_check:
        raise ValueError(
            'model not supported, please check supported models from `malaya.similarity.semantic.available_huggingface()`.'
        )
    return load_huggingface.load_similarity(model=model, **kwargs)
