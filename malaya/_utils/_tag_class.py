import pickle
import json
from ._utils import check_file, load_graph, check_available, generate_session
from .._models._tensorflow_model import TAGGING
from .._models._bert_model import TAGGING_BERT
from .._models._sklearn_model import CRF


def crf(path, s3_path, class_name, validate = True):
    if validate:
        check_file(path['crf'], s3_path['crf'])
    else:
        if not check_available(path['crf']):
            raise Exception(
                '%s/crf is not available, please `validate = True`'
                % (class_name)
            )
    try:
        with open(path['crf']['model'], 'rb') as fopen:
            model = pickle.load(fopen)
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/crf') and try again"
            % (class_name)
        )
    return CRF(model)


def deep_model(path, s3_path, class_name, model = 'bahdanau', validate = True):
    if not isinstance(model, str):
        raise ValueError('model must be a string')

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
            nodes = json.loads(fopen.read())
        g = load_graph(path[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
            % (class_name, model)
        )
    return TAGGING(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/Placeholder_1:0'),
        g.get_tensor_by_name('import/logits:0'),
        nodes,
        generate_session(graph = g),
        model,
        g.get_tensor_by_name('import/transitions:0'),
        g.get_tensor_by_name('import/Variable:0'),
    )


def bert(path, s3_path, class_name, model = 'base', validate = True):
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
        with open(path[model]['setting'], 'r') as fopen:
            nodes = json.loads(fopen.read())
        g = load_graph(path[model]['model'])
    except:
        raise Exception(
            "model corrupted due to some reasons, please run malaya.clear_cache('%s/%s') and try again"
            % (class_name, model)
        )

    return TAGGING_BERT(
        X = g.get_tensor_by_name('import/Placeholder:0'),
        segment_ids = None,
        input_masks = None,
        logits = g.get_tensor_by_name('import/logits:0'),
        sess = generate_session(graph = g),
        tokenizer = tokenizer,
        cls = cls,
        sep = sep,
        settings = nodes,
    )
