import json
import os
import pickle
from ._utils import check_file, load_graph, check_available, generate_session
from .._models._sklearn_model import (
    BINARY_XGB,
    BINARY_BAYES,
    MULTICLASS_XGB,
    MULTICLASS_BAYES,
)
from .._models._tensorflow_model import BINARY_SOFTMAX, MULTICLASS_SOFTMAX
from .._models._bert_model import MULTICLASS_BERT, BINARY_BERT
from ..bert import _extract_attention_weights_import, bert_num_layers


def deep_model(
    path, s3_path, class_name, label, model = 'luong', validate = True
):
    if validate:
        check_file(path[model], s3_path[model])
    else:
        if not check_available(path[model]):
            raise Exception(
                '%s/%s is not available, please `validate = True`'
                % (class_name, model)
            )
    try:
        with open(path[model]['setting'], 'r') as fopen:
            dictionary = json.load(fopen)['dictionary']
        g = load_graph(path[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
            % (class_name, model)
        )

    if len(label) > 2 or class_name == 'relevancy':
        selected_class = MULTICLASS_SOFTMAX
    else:
        selected_class = BINARY_SOFTMAX

    return selected_class(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        logits = g.get_tensor_by_name('import/logits:0'),
        logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
        alphas = g.get_tensor_by_name('import/alphas:0'),
        sess = generate_session(graph = g),
        dictionary = dictionary,
        class_name = class_name,
        label = label,
    )


def multinomial(path, s3_path, class_name, label, validate = True):
    if validate:
        check_file(path['multinomial'], s3_path['multinomial'])
    else:
        if not check_available(path['multinomial']):
            raise Exception(
                '%s/multinomial is not available, please `validate = True`'
                % (class_name)
            )
    try:
        with open(path['multinomial']['model'], 'rb') as fopen:
            multinomial = pickle.load(fopen)
        with open(path['multinomial']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/multinomial') and try again"
            % (class_name)
        )
    from ..stem import _classification_textcleaning_stemmer

    if len(label) > 2:
        selected_class = MULTICLASS_BAYES
    else:
        selected_class = BINARY_BAYES
    return selected_class(
        multinomial = multinomial,
        label = label,
        vectorize = vectorize,
        cleaning = _classification_textcleaning_stemmer,
    )


def xgb(path, s3_path, class_name, label, validate = True):
    if validate:
        check_file(path['xgb'], s3_path['xgb'])
    else:
        if not check_available(path['xgb']):
            raise Exception(
                '%s/xgb is not available, please `validate = True`'
                % (class_name)
            )
    try:
        with open(path['xgb']['model'], 'rb') as fopen:
            xgb = pickle.load(fopen)
        with open(path['xgb']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/xgb') and try again"
            % (class_name)
        )

    from ..stem import _classification_textcleaning_stemmer

    if len(label) > 2:
        selected_class = MULTICLASS_XGB
    else:
        selected_class = BINARY_XGB

    return selected_class(
        xgb = xgb,
        label = label,
        vectorize = vectorize,
        cleaning = _classification_textcleaning_stemmer,
    )


def bert(path, s3_path, class_name, label, model = 'base', validate = True):
    if validate:
        check_file(path[model], s3_path[model])
    else:
        if not check_available(path[model]):
            raise Exception(
                '%s/%s is not available, please `validate = True`'
                % (class_name, model)
            )
    if model == 'multilanguage':
        from bert import tokenization

        tokenizer = tokenization.FullTokenizer(
            vocab_file = path[model]['vocab'], do_lower_case = False
        )
        cls = '[CLS]'
        sep = '[SEP]'
    else:

        import sentencepiece as spm
        from ..texts._text_functions import SentencePieceTokenizer

        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(path[model]['tokenizer'])

        with open(path[model]['vocab']) as fopen:
            v = fopen.read().split('\n')[:-1]
        v = [i.split('\t') for i in v]
        v = {i[0]: i[1] for i in v}
        tokenizer = SentencePieceTokenizer(v, sp_model)
        cls = '<cls>'
        sep = '<sep>'

    try:
        g = load_graph(path[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
            % (class_name, model)
        )

    if len(label) > 2 or class_name == 'relevancy':
        selected_class = MULTICLASS_BERT
    else:
        selected_class = BINARY_BERT

    return selected_class(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        segment_ids = None,
        input_masks = None,
        logits = g.get_tensor_by_name('import/logits:0'),
        logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
        sess = generate_session(graph = g),
        tokenizer = tokenizer,
        label = label,
        cls = cls,
        sep = sep,
        attns = _extract_attention_weights_import(bert_num_layers[model], g),
        class_name = class_name,
    )
