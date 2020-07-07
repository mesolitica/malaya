from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import (
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
)
from malaya.path import PATH_TOXIC, S3_PATH_TOXIC
from malaya.model.ml import MULTILABEL_BAYES
from malaya.model.bert import SIGMOID_BERT
from malaya.model.xlnet import SIGMOID_XLNET
from malaya.transformers.bert import bert_num_layers
from herpetologist import check_type

label = [
    'severe toxic',
    'obscene',
    'identity attack',
    'insult',
    'threat',
    'asian',
    'atheist',
    'bisexual',
    'buddhist',
    'christian',
    'female',
    'heterosexual',
    'indian',
    'homosexual, gay or lesbian',
    'intellectual or learning disability',
    'male',
    'muslim',
    'other disability',
    'other gender',
    'other race or ethnicity',
    'other religion',
    'other sexual orientation',
    'physical disability',
    'psychiatric or mental illness',
    'transgender',
    'malay',
    'chinese',
]


_availability = {
    'bert': ['425.7 MB', 'accuracy: 0.814'],
    'tiny-bert': ['57.4 MB', 'accuracy: 0.815'],
    'albert': ['48.7 MB', 'accuracy: 0.812'],
    'tiny-albert': ['22.4 MB', 'accuracy: 0.808'],
    'xlnet': ['446.5 MB', 'accuracy: 0.807'],
    'alxlnet': ['46.8 MB', 'accuracy: 0.817'],
}


def available_transformer():
    """
    List available transformer toxicity analysis models.
    """
    return _availability


def multinomial(**kwargs):
    """
    Load multinomial toxicity model.

    Returns
    -------
    result : malaya.model.ml.MULTILABEL_BAYES class
    """
    import pickle

    check_file(
        PATH_TOXIC['multinomial'], S3_PATH_TOXIC['multinomial'], **kwargs
    )

    try:
        with open(PATH_TOXIC['multinomial']['model'], 'rb') as fopen:
            multinomial = pickle.load(fopen)
        with open(PATH_TOXIC['multinomial']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            f"model corrupted due to some reasons, please run malaya.clear_cache('toxic/multinomial') and try again"
        )

    from malaya.text.bpe import load_yttm
    from malaya.stem import _classification_textcleaning_stemmer

    bpe, subword_mode = load_yttm(PATH_TOXIC['multinomial']['bpe'])

    return MULTILABEL_BAYES(
        multinomial = multinomial,
        label = label,
        vectorize = vectorize,
        bpe = bpe,
        subword_mode = subword_mode,
        cleaning = _classification_textcleaning_stemmer,
    )


@check_type
def transformer(model: str = 'xlnet', **kwargs):
    """
    Load Transformer toxicity model.

    Parameters
    ----------
    model : str, optional (default='bert')
        Model architecture supported. Allowed values:

        * ``'bert'`` - BERT architecture from google.
        * ``'tiny-bert'`` - BERT architecture from google with smaller parameters.
        * ``'albert'`` - ALBERT architecture from google.
        * ``'tiny-albert'`` - ALBERT architecture from google with smaller parameters.
        * ``'xlnet'`` - XLNET architecture from google.
        * ``'alxlnet'`` - XLNET architecture from google + Malaya.

    Returns
    -------
    result : malaya.model.bert.SIGMOID_BERT class
    """

    model = model.lower()
    if model not in _availability:
        raise Exception(
            'model not supported, please check supported models from malaya.sentiment.available_transformer()'
        )

    check_file(PATH_TOXIC[model], S3_PATH_TOXIC[model], **kwargs)
    g = load_graph(PATH_TOXIC[model]['model'])

    path = PATH_TOXIC

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        if model in ['bert', 'tiny-bert']:
            from malaya.transformers.bert import (
                _extract_attention_weights_import,
            )
            from malaya.transformers.bert import bert_num_layers

            tokenizer = sentencepiece_tokenizer_bert(
                path[model]['tokenizer'], path[model]['vocab']
            )
        if model in ['albert', 'tiny-albert']:
            from malaya.transformers.albert import (
                _extract_attention_weights_import,
            )
            from malaya.transformers.albert import bert_num_layers
            from albert import tokenization

            tokenizer = tokenization.FullTokenizer(
                vocab_file = path[model]['vocab'],
                do_lower_case = False,
                spm_model_file = path[model]['tokenizer'],
            )

        return SIGMOID_BERT(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = None,
            input_masks = g.get_tensor_by_name('import/Placeholder_1:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            label = label,
            attns = _extract_attention_weights_import(
                bert_num_layers[model], g
            ),
            class_name = 'toxic',
        )

    if model in ['xlnet', 'alxlnet']:
        if model in ['xlnet']:
            from malaya.transformers.xlnet import (
                _extract_attention_weights_import,
            )
        if model in ['alxlnet']:
            from malaya.transformers.alxlnet import (
                _extract_attention_weights_import,
            )

        tokenizer = sentencepiece_tokenizer_xlnet(path[model]['tokenizer'])

        return SIGMOID_XLNET(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            label = label,
            attns = _extract_attention_weights_import(g),
            class_name = 'toxic',
        )
