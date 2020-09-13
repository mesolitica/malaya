from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import SentencePieceEncoder
import tensorflow as tf
import os


def load(path, s3_path, model, model_class, **kwargs):
    check_file(path[model], s3_path[model], **kwargs)
    g = load_graph(path[model]['model'], **kwargs)
    X = g.get_tensor_by_name('import/inputs:0')

    tokenizer = SentencePieceEncoder(path[model]['tokenizer'])
