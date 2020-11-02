from malaya.function import check_file, load_graph, generate_session
from malaya.text.bpe import SentencePieceEncoder, YTTMEncoder, load_yttm
from malaya.text.t2t import text_encoder


def load_lm(path, s3_path, model, model_class, quantized = False, **kwargs):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)
    X = g.get_tensor_by_name('import/Placeholder:0')
    top_p = g.get_tensor_by_name('import/Placeholder_2:0')
    greedy = g.get_tensor_by_name('import/greedy:0')
    beam = g.get_tensor_by_name('import/beam:0')
    nucleus = g.get_tensor_by_name('import/nucleus:0')

    tokenizer = SentencePieceEncoder(path[model]['vocab'])

    return model_class(
        X,
        top_p,
        greedy,
        beam,
        nucleus,
        generate_session(graph = g, **kwargs),
        tokenizer,
    )


def load(
    path, s3_path, model, encoder, model_class, quantized = False, **kwargs
):
    check_file(path[model], s3_path[model], quantized = quantized, **kwargs)
    if quantized:
        model_path = 'quantized'
    else:
        model_path = 'model'

    g = load_graph(path[model][model_path], **kwargs)

    if encoder == 'subword':
        encoder = text_encoder.SubwordTextEncoder(path[model]['vocab'])

    if encoder == 'yttm':
        bpe, subword_mode = load_yttm(path[model]['vocab'], True)
        encoder = YTTMEncoder(bpe, subword_mode)

    return model_class(
        g.get_tensor_by_name('import/Placeholder:0'),
        g.get_tensor_by_name('import/greedy:0'),
        g.get_tensor_by_name('import/beam:0'),
        generate_session(graph = g, **kwargs),
        encoder,
    )
