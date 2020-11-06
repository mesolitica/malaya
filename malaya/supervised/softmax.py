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
    path,
    s3_path,
    class_name,
    label,
    model = 'bert',
    quantized = False,
    **kwargs,
):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'
    g = load_graph(path[model][model_path], **kwargs)

    if len(label) > 2 or class_name == 'relevancy':
        if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
            selected_class = MULTICLASS_BERT
            selected_node = 'import/dense/BiasAdd:0'
        if model in ['xlnet', 'alxlnet']:
            selected_class = MULTICLASS_XLNET
            selected_node = 'import/transpose_3:0'

    else:
        if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
            selected_class = BINARY_BERT
            selected_node = 'import/dense/BiasAdd:0'
        if model in ['xlnet', 'alxlnet']:
            selected_class = BINARY_XLNET
            selected_node = 'import/transpose_3:0'

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

        return selected_class(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = None,
            input_masks = g.get_tensor_by_name('import/Placeholder_1:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
            vectorizer = g.get_tensor_by_name(selected_node),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            label = label,
            attns = _extract_attention_weights_import(
                bert_num_layers[model], g
            ),
            class_name = class_name,
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

        return selected_class(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
            vectorizer = g.get_tensor_by_name(selected_node),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            label = label,
            attns = _extract_attention_weights_import(g),
            class_name = class_name,
        )
