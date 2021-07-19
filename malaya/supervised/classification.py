from malaya.stem import _classification_textcleaning_stemmer, naive
from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import (
    WordPieceTokenizer,
    SentencePieceTokenizer,
    YTTMEncoder,
)
from malaya.model.ml import BinaryBayes, MulticlassBayes, MultilabelBayes
from malaya.model.bert import MulticlassBERT, BinaryBERT, SigmoidBERT
from malaya.model.xlnet import MulticlassXLNET, BinaryXLNET, SigmoidXLNET
from malaya.model.fnet import MulticlassFNet, BinaryFNet
from malaya.model.bigbird import MulticlassBigBird
from malaya.transformers.bert import bert_num_layers
from malaya.transformers.albert import albert_num_layers
from malaya.transformers.bert import _extract_attention_weights_import as bert_attention_weights
from malaya.transformers.albert import _extract_attention_weights_import as albert_attention_weights
from malaya.transformers.alxlnet import _extract_attention_weights_import as alxlnet_attention_weights
from malaya.transformers.xlnet import _extract_attention_weights_import as xlnet_attention_weights
from malaya.path import MODEL_VOCAB, MODEL_BPE
from functools import partial
import json
import os
import pickle
import numpy as np

SIGMOID_MODEL = {
    'albert': SigmoidBERT,
    'bert': SigmoidBERT,
    'tiny-albert': SigmoidBERT,
    'tiny-bert': SigmoidBERT,
    'xlnet': SigmoidXLNET,
    'alxlnet': SigmoidXLNET,
}
MULTICLASS_MODEL = {
    'albert': MulticlassBERT,
    'tiny-albert': MulticlassBERT,
    'bert': MulticlassBERT,
    'tiny-bert': MulticlassBERT,
    'alxlnet': MulticlassXLNET,
    'xlnet': MulticlassXLNET,
    'bigbird': MulticlassBigBird,
    'tiny-bigbird': MulticlassBigBird,
    'fnet': MulticlassFNet,
    'large-fnet': MulticlassFNet,
}
BINARY_MODEL = {
    'albert': BinaryBERT,
    'tiny-albert': BinaryBERT,
    'bert': BinaryBERT,
    'tiny-bert': BinaryBERT,
    'alxlnet': BinaryXLNET,
    'xlnet': BinaryXLNET,
    'fnet': BinaryFNet,
    'large-fnet': BinaryFNet,
}
TOKENIZER_MODEL = {
    'bert': SentencePieceTokenizer,
    'tiny-bert': SentencePieceTokenizer,
    'albert': SentencePieceTokenizer,
    'tiny-albert': SentencePieceTokenizer,
    'bigbird': SentencePieceTokenizer,
    'tiny-bigbird': SentencePieceTokenizer,
    'fnet': WordPieceTokenizer,
    'large-fnet': WordPieceTokenizer,
    'alxlnet': SentencePieceTokenizer,
    'xlnet': SentencePieceTokenizer,
}


def multinomial(path, s3_path, module, label, sigmoid=False, **kwargs):
    check_file(path['multinomial'], s3_path['multinomial'], **kwargs)
    try:
        with open(path['multinomial']['model'], 'rb') as fopen:
            multinomial = pickle.load(fopen)
        with open(path['multinomial']['vector'], 'rb') as fopen:
            vectorize = pickle.load(fopen)
    except BaseException:
        path = os.path.normpath(f'{module}/multinomial')
        raise Exception(
            f"model corrupted due to some reasons, please run `malaya.clear_cache('{path}')` and try again"
        )

    bpe = YTTMEncoder(vocab_file=path['multinomial']['bpe'])

    stemmer = naive()
    cleaning = partial(_classification_textcleaning_stemmer, stemmer=stemmer)

    if sigmoid:
        selected_model = MultilabelBayes
    else:
        if len(label) > 2:
            selected_model = MulticlassBayes
        else:
            selected_model = BinaryBayes
    return selected_model(
        multinomial=multinomial,
        label=label,
        vectorize=vectorize,
        bpe=bpe,
        cleaning=cleaning,
    )


def transformer(
    module,
    label,
    model='bert',
    sigmoid=False,
    quantized=False,
    **kwargs,
):
    path = check_file(
        file=model,
        module=module,
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
        selected_model = SIGMOID_MODEL[model]
    else:
        if len(label) > 2 or module == 'relevancy':
            selected_model = MULTICLASS_MODEL[model]
        else:
            selected_model = BINARY_MODEL[model]

    if model in ['albert', 'bert', 'tiny-albert', 'tiny-bert']:
        if model in ['bert', 'tiny-bert']:
            attention = bert_attention_weights(bert_num_layers[model], g)

        if model in ['albert', 'tiny-albert']:
            attention = albert_attention_weights(albert_num_layers[model], g)

        inputs = ['Placeholder', 'Placeholder_1']
        vectorizer = {'vectorizer': 'import/dense/BiasAdd:0'}

    if model in ['xlnet', 'alxlnet']:
        if model in ['xlnet']:
            weights_import = xlnet_attention_weights
        if model in ['alxlnet']:
            weights_import = alxlnet_attention_weights

        inputs = ['Placeholder', 'Placeholder_1', 'Placeholder_2']
        vectorizer = {'vectorizer': 'import/transpose_3:0'}
        attention = weights_import(g)

    if model in ['bigbird', 'tiny-bigbird']:
        inputs = ['Placeholder']
        vectorizer = {'vectorizer': 'import/dense/BiasAdd:0'}
        attention = None

    if model in ['fnet', 'fnet-large']:
        inputs = ['Placeholder', 'Placeholder_1']
        vectorizer = {'vectorizer': 'import/vectorizer:0'}
        attention = None

    outputs = ['logits', 'logits_seq']
    tokenizer = TOKENIZER_MODEL[model](vocab_file=path['vocab'], spm_model_file=path['tokenizer'])
    input_nodes, output_nodes = nodes_session(
        g,
        inputs,
        outputs,
        extra=vectorizer,
        attention={'attention': attention},
    )

    return selected_model(
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        label=label,
        module=module,
    )
