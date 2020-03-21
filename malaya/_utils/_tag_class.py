import json
from ._utils import (
    check_file,
    load_graph,
    generate_session,
    sentencepiece_tokenizer_bert,
    sentencepiece_tokenizer_xlnet,
)
from .._models._bert_model import TAGGING_BERT
from .._models._xlnet_model import TAGGING_XLNET


def transformer(
    path, s3_path, class_name, model = 'xlnet', size = 'base', **kwargs
):
    check_file(path[model][size], s3_path[model][size], **kwargs)

    try:
        with open(path[model][size]['setting']) as fopen:
            nodes = json.load(fopen)
        g = load_graph(path[model][size]['model'])
    except:
        raise Exception(
            f"model corrupted due to some reasons, please run malaya.clear_cache('{class_name}/{model}/{size}') and try again"
        )

    if model in ['albert', 'bert']:
        tokenizer, cls, sep = sentencepiece_tokenizer_bert(
            path[model][size]['tokenizer'], path[model][size]['vocab']
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

    if model in ['xlnet']:
        tokenizer = sentencepiece_tokenizer_xlnet(
            path[model][size]['tokenizer']
        )
        return TAGGING_XLNET(
            X = g.get_tensor_by_name('import/Placeholder:0'),
            segment_ids = g.get_tensor_by_name('import/Placeholder_1:0'),
            input_masks = g.get_tensor_by_name('import/Placeholder_2:0'),
            logits = g.get_tensor_by_name('import/logits:0'),
            sess = generate_session(graph = g),
            tokenizer = tokenizer,
            settings = nodes,
        )
