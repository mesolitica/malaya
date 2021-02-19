import json
import os
import pickle
import numpy as np
from functools import partial
from malaya.stem import _classification_textcleaning_stemmer, naive
from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import (
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
    load_yttm,
)
from malaya.model.ml import BinaryBayes, MulticlassBayes, MultilabelBayes
from malaya.model.bert import MulticlassBERT, BinaryBERT, SigmoidBERT
from malaya.model.xlnet import MulticlassXLNET, BinaryXLNET, SigmoidXLNET
from malaya.model.bigbird import MulticlassBigBird
from malaya.path import MODEL_VOCAB, MODEL_BPE


def multinomial(path, s3_path, class_name, label, sigmoid = False, **kwargs):
    check_file(path['multinomial'], s3_path['multinomial'], **kwargs)
    try:
        with open(path['multinomial']['model'], 'rb') as fopen:
            multinomial = pickle.load(fopen)
        with open(path['multinomial']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except:
        raise Exception(
            f"model corrupted due to some reasons, please run `malaya.clear_cache('{class_name}/multinomial')` and try again"
        )
    bpe, subword_mode = load_yttm(path['multinomial']['bpe'])

    stemmer = naive()
    cleaning = partial(_classification_textcleaning_stemmer, stemmer = stemmer)

    if sigmoid:
        selected_class = MultilabelBayes
    else:
        if len(label) > 2:
            selected_class = MulticlassBayes
        else:
            selected_class = BinaryBayes
    return selected_class(
        multinomial = multinomial,
        label = label,
        vectorize = vectorize,
        bpe = bpe,
        subword_mode = subword_mode,
        cleaning = cleaning,
    )


def transformer(
    class_name,
    label,
    model = 'bert',
    sigmoid = False,
    quantized = False,
    **kwargs,
):
    path = check_file(
        file = model,
        module = class_name,
        keys = {
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
        },
        quantized = quantized,
        **kwargs,
    )
    g = load_graph(path['model'], **kwargs)

    if sigmoid:
        if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
            selected_class = SigmoidBERT
        if model in ['xlnet', 'alxlnet']:
            selected_class = SigmoidXLNET
    else:
        if len(label) > 2 or class_name == 'relevancy':
            if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
                selected_class = MulticlassBERT
            if model in ['xlnet', 'alxlnet']:
                selected_class = MulticlassXLNET
            if model in ['bigbird', 'tiny-bigbird']:
                selected_class = MulticlassBigBird

        else:
            if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
                selected_class = BinaryBERT
            if model in ['xlnet', 'alxlnet']:
                selected_class = BinaryXLNET

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        selected_node = 'import/dense/BiasAdd:0'
        if model in ['bert', 'tiny-bert']:
            from malaya.transformers.bert import (
                _extract_attention_weights_import,
            )
            from malaya.transformers.bert import bert_num_layers

            tokenizer = sentencepiece_tokenizer_bert(
                path['tokenizer'], path['vocab']
            )
        if model in ['albert', 'tiny-albert']:
            from malaya.transformers.albert import (
                _extract_attention_weights_import,
            )
            from malaya.transformers.albert import bert_num_layers
            from malaya.transformers.albert import tokenization

            tokenizer = tokenization.FullTokenizer(
                vocab_file = path['vocab'],
                do_lower_case = False,
                spm_model_file = path['tokenizer'],
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
        selected_node = 'import/transpose_3:0'
        if model in ['xlnet']:
            from malaya.transformers.xlnet import (
                _extract_attention_weights_import,
            )
        if model in ['alxlnet']:
            from malaya.transformers.alxlnet import (
                _extract_attention_weights_import,
            )

        tokenizer = sentencepiece_tokenizer_xlnet(path['tokenizer'])

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

    if model in ['bigbird', 'tiny-bigbird']:
        selected_node = 'import/dense/BiasAdd:0'
        tokenizer = sentencepiece_tokenizer_bert(
            path['tokenizer'], path['vocab']
        )
        return selected_class(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            logits_seq = g.get_tensor_by_name('import/logits_seq:0'),
            vectorizer = g.get_tensor_by_name(selected_node),
            sess = generate_session(graph = g, **kwargs),
            tokenizer = tokenizer,
            label = label,
            class_name = class_name,
        )
