import json
import re
from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import (
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
)
from malaya.model.bert import TaggingBERT
from malaya.model.xlnet import TaggingXLNET
from malaya.text.regex import _expressions
from malaya.path import MODEL_VOCAB, MODEL_BPE, TAGGING_SETTING


def transformer(class_name, model = 'xlnet', quantized = False, **kwargs):
    path = check_file(
        file = model,
        module = class_name,
        keys = {
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
            'setting': TAGGING_SETTING[class_name],
        },
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    try:
        with open(path['setting']) as fopen:
            nodes = json.load(fopen)
    except:
        raise Exception(
            f"model corrupted due to some reasons, please run `malaya.clear_cache('{class_name}/{model}/{size}')` and try again"
        )

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        if model in ['bert', 'tiny-bert']:
            tokenizer = sentencepiece_tokenizer_bert(
                path['tokenizer'], path['vocab']
            )

        if model in ['albert', 'tiny-albert']:
            from malaya.transformers.albert import tokenization

            tokenizer = tokenization.FullTokenizer(
                vocab_file = path['vocab'],
                do_lower_case = False,
                spm_model_file = path['tokenizer'],
            )

        return TaggingBERT(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = None,
            input_masks = g.get_tensor_by_name('import/Placeholder_1:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            vectorizer = g.get_tensor_by_name('import/dense/BiasAdd:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            settings = nodes,
        )

    if model in ['xlnet', 'alxlnet']:
        tokenizer = sentencepiece_tokenizer_xlnet(path['tokenizer'])
        return TaggingXLNET(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            vectorizer = g.get_tensor_by_name('import/transpose_3:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            settings = nodes,
        )


def transformer_ontonotes5(
    class_name, model = 'xlnet', quantized = False, **kwargs
):
    path = check_file(
        file = model,
        module = class_name,
        keys = {
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
            'setting': TAGGING_SETTING[class_name],
        },
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    hypen = r'\w+(?:-\w+)+'
    hypen_left = r'\w+(?: -\w+)+'
    hypen_right = r'\w+(?:- \w+)+'
    hypen_both = r'\w+(?: - \w+)+'

    pipeline = [
        hypen,
        hypen_left,
        hypen_right,
        hypen_both,
        _expressions['percent'],
        _expressions['money'],
        _expressions['time'],
        _expressions['date'],
        _expressions['repeat_puncts'],
        _expressions['number'],
        _expressions['word'],
    ]
    pipeline.append('(?:\S)')
    compiled = re.compile(r'({})'.format('|'.join(pipeline)))

    def tok(string):
        tokens = compiled.findall(string)
        return [t[0] for t in tokens]

    try:
        with open(path['setting']) as fopen:
            nodes = json.load(fopen)
    except:
        raise Exception(
            f"model corrupted due to some reasons, please run `malaya.clear_cache('{class_name}/{model}/{size}')` and try again"
        )

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        if model in ['bert', 'tiny-bert']:
            tokenizer = sentencepiece_tokenizer_bert(
                path['tokenizer'], path['vocab']
            )

        if model in ['albert', 'tiny-albert']:
            from malaya.transformers.albert import tokenization

            tokenizer = tokenization.FullTokenizer(
                vocab_file = path['vocab'],
                do_lower_case = False,
                spm_model_file = path['tokenizer'],
            )

        return TaggingBERT(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = None,
            input_masks = g.get_tensor_by_name('import/Placeholder_1:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            vectorizer = g.get_tensor_by_name('import/dense/BiasAdd:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            settings = nodes,
            tok = tok,
        )

    if model in ['xlnet', 'alxlnet']:
        tokenizer = sentencepiece_tokenizer_xlnet(path['tokenizer'])
        return TaggingXLNET(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            vectorizer = g.get_tensor_by_name('import/transpose_3:0'),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            settings = nodes,
            tok = tok,
        )
