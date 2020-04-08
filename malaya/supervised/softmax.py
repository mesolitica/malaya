import json
import os
import pickle
import numpy as np
from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import (
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
    load_yttm,
)
from malaya.model.ml import BINARY_BAYES, MULTICLASS_BAYES
from malaya.model.bert import MULTICLASS_BERT, BINARY_BERT
from malaya.model.xlnet import MULTICLASS_XLNET, BINARY_XLNET
from malaya.transformer.bert import bert_num_layers


def multinomial(path, s3_path, class_name, label, **kwargs):
    check_file(path['multinomial'], s3_path['multinomial'], **kwargs)
    try:
        with open(path['multinomial']['model'], 'rb') as fopen:
            multinomial = pickle.load(fopen)
        with open(path['multinomial']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            f"model corrupted due to some reasons, please run malaya.clear_cache('{class_name}/multinomial') and try again"
        )
    bpe, subword_mode = load_yttm(path['multinomial']['bpe'])

    from malaya.stem import _classification_textcleaning_stemmer

    if len(label) > 2:
        selected_class = MULTICLASS_BAYES
    else:
        selected_class = BINARY_BAYES
    return selected_class(
        multinomial = multinomial,
        label = label,
        vectorize = vectorize,
        bpe = bpe,
        subword_mode = subword_mode,
        cleaning = _classification_textcleaning_stemmer,
    )


def transformer(
    path, s3_path, class_name, label, model = 'bert', size = 'base', **kwargs
):
    check_file(path[model][size], s3_path[model][size], **kwargs)
    g = load_graph(path[model][size]['model'])

    if len(label) > 2 or class_name == 'relevancy':
        if model in ['albert', 'bert']:
            selected_class = MULTICLASS_BERT
        if model in ['xlnet']:
            selected_class = MULTICLASS_XLNET

    else:
        if model in ['albert', 'bert']:
            selected_class = BINARY_BERT
        if model in ['xlnet']:
            selected_class = BINARY_XLNET

    if model in ['albert', 'bert']:
        if model == 'bert':
            from .._transformer._bert import _extract_attention_weights_import
        if model == 'albert':
            from .._transformer._albert import _extract_attention_weights_import

        tokenizer = sentencepiece_tokenizer_bert(
            path[model]['tokenizer'], path[model]['vocab']
        )

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
            attns = _extract_attention_weights_import(bert_num_layers[size], g),
            class_name = class_name,
        )
    if model in ['xlnet']:
        from .._transformer._xlnet import _extract_attention_weights_import

        tokenizer = sentencepiece_tokenizer_xlnet(
            path[model][size]['tokenizer']
        )

        return selected_class(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
            sess = generate_session(graph = g),
            tokenizer = tokenizer,
            label = label,
            attns = _extract_attention_weights_import(g),
            class_name = class_name,
        )
