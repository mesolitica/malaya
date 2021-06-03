import json
import os
import pickle
import numpy as np
from functools import partial
from malaya.stem import _classification_textcleaning_stemmer, naive
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import (
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
    WordPieceTokenizer,
    AlbertTokenizer,
    load_yttm,
)
from malaya.model.ml import BinaryBayes, MulticlassBayes, MultilabelBayes
from malaya.model.bert import MulticlassBERT, BinaryBERT, SigmoidBERT
from malaya.model.xlnet import MulticlassXLNET, BinaryXLNET, SigmoidXLNET
from malaya.model.fnet import MulticlassFNet, BinaryFNet
from malaya.model.bigbird import MulticlassBigBird
from malaya.path import MODEL_VOCAB, MODEL_BPE


def multinomial(path, s3_path, class_name, label, sigmoid=False, **kwargs):
    check_file(path['multinomial'], s3_path['multinomial'], **kwargs)
    try:
        with open(path['multinomial']['model'], 'rb') as fopen:
            multinomial = pickle.load(fopen)
        with open(path['multinomial']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except BaseException:
        raise Exception(
            f"model corrupted due to some reasons, please run `malaya.clear_cache('{class_name}/multinomial')` and try again"
        )
    bpe, subword_mode = load_yttm(path['multinomial']['bpe'])

    stemmer = naive()
    cleaning = partial(_classification_textcleaning_stemmer, stemmer=stemmer)

    if sigmoid:
        selected_class = MultilabelBayes
    else:
        if len(label) > 2:
            selected_class = MulticlassBayes
        else:
            selected_class = BinaryBayes
    return selected_class(
        multinomial=multinomial,
        label=label,
        vectorize=vectorize,
        bpe=bpe,
        subword_mode=subword_mode,
        cleaning=cleaning,
    )


def transformer(
    class_name,
    label,
    model='bert',
    sigmoid=False,
    quantized=False,
    **kwargs,
):
    path = check_file(
        file=model,
        module=class_name,
        keys={
            'model': 'model.pb',
            'vocab': MODEL_VOCAB[model],
            'tokenizer': MODEL_BPE[model],
        },
        quantized=quantized,
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
            if model in ['fnet', 'fnet-large']:
                selected_class = MulticlassFNet

        else:
            if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
                selected_class = BinaryBERT
            if model in ['xlnet', 'alxlnet']:
                selected_class = BinaryXLNET
            if model in ['fnet', 'fnet-large']:
                selected_class = BinaryFNet

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
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

            tokenizer = AlbertTokenizer(
                vocab_file=path['vocab'], spm_model_file=path['tokenizer']
            )

        inputs = ['Placeholder', 'Placeholder_1']
        vectorizer = {'vectorizer': 'import/dense/BiasAdd:0'}
        attention = _extract_attention_weights_import(bert_num_layers[model], g)

    if model in ['xlnet', 'alxlnet']:
        if model in ['xlnet']:
            from malaya.transformers.xlnet import (
                _extract_attention_weights_import,
            )
        if model in ['alxlnet']:
            from malaya.transformers.alxlnet import (
                _extract_attention_weights_import,
            )

        inputs = ['Placeholder', 'Placeholder_1', 'Placeholder_2']
        tokenizer = sentencepiece_tokenizer_xlnet(path['tokenizer'])
        vectorizer = {'vectorizer': 'import/transpose_3:0'}
        attention = _extract_attention_weights_import(g)

    if model in ['bigbird', 'tiny-bigbird']:
        inputs = ['Placeholder']
        tokenizer = sentencepiece_tokenizer_bert(
            path['tokenizer'], path['vocab']
        )
        vectorizer = {'vectorizer': 'import/dense/BiasAdd:0'}
        attention = None

    if model in ['fnet', 'fnet-large']:
        inputs = ['Placeholder', 'Placeholder_1']
        tokenizer = WordPieceTokenizer(path['tokenizer'])
        vectorizer = {'vectorizer': 'import/vectorizer:0'}
        attention = None

    outputs = ['logits', 'logits_seq']
    input_nodes, output_nodes = nodes_session(
        g,
        inputs,
        outputs,
        extra=vectorizer,
        attention={'attention': attention},
    )

    return selected_class(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        label=label,
        class_name=class_name,
    )
