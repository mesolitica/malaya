from malaya.function import (
    check_file,
    load_graph,
    generate_session,
    nodes_session,
)
from malaya.text.bpe import SentencePieceBatchEncoder
from malaya.path import MS_EN_BPE_MODEL, MS_EN_4k_BPE_MODEL, T2T_BPE_MODEL
from malaya.preprocessing import Tokenizer

VOCAB_MODEL = {'generator': T2T_BPE_MODEL}


def load(module, model, model_class, quantized=False, **kwargs):

    try:
        import tensorflow_text
    except BaseException:
        raise ModuleNotFoundError(
            'tensorflow-text not installed. Please install it by `pip install tensorflow-text` and try again. Also, make sure tensorflow-text version same as tensorflow version.'
        )

    if model.split('-')[-1] == '4k':
        default_vocab = MS_EN_4k_BPE_MODEL
    else:
        default_vocab = MS_EN_BPE_MODEL

    path = check_file(
        file=model,
        module=module,
        keys={'model': 'model.pb', 'vocab': VOCAB_MODEL.get(module, default_vocab)},
        quantized=quantized,
        **kwargs,
    )
    g = load_graph(path['model'], t5_graph=True, **kwargs)
    tokenizer = SentencePieceBatchEncoder(vocab_file=path['vocab'])
    inputs = ['inputs']
    outputs = []
    input_nodes, output_nodes = nodes_session(
        g, inputs, outputs, extra={'decode': 'import/SelectV2_3:0'}
    )

    if module == 'kesalahan-tatabahasa':
        word_tokenizer = Tokenizer(date=False, time=False).tokenize
    elif module == 'spelling-correction':
        word_tokenizer = Tokenizer(duration=False, date=False).tokenize
    else:
        word_tokenizer = None

    return model_class(
        input_nodes=input_nodes, output_nodes=output_nodes,
        sess=generate_session(graph=g, **kwargs),
        tokenizer=tokenizer,
        word_tokenizer=word_tokenizer,
    )
