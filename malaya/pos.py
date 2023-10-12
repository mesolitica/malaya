import re
from malaya.text.tatabahasa import tatabahasa_dict, hujung, permulaan
from malaya.supervised.huggingface import load
from malaya.torch_model.huggingface import Tagging

label = [
    'OTHER',
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CCONJ',
    'DET',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X'
]


describe = [
    {'Tag': 'ADJ', 'Description': 'Adjective, kata sifat'},
    {'Tag': 'ADP', 'Description': 'Adposition'},
    {'Tag': 'ADV', 'Description': 'Adverb, kata keterangan'},
    {'Tag': 'ADX', 'Description': 'Auxiliary verb, kata kerja tambahan'},
    {'Tag': 'CCONJ', 'Description': 'Coordinating conjuction, kata hubung'},
    {'Tag': 'DET', 'Description': 'Determiner, kata penentu'},
    {'Tag': 'NOUN', 'Description': ' Noun, kata nama'},
    {'Tag': 'NUM', 'Description': 'Number, nombor'},
    {'Tag': 'PART', 'Description': 'Particle'},
    {'Tag': 'PRON', 'Description': 'Pronoun, kata ganti'},
    {'Tag': 'PROPN', 'Description': 'Proper noun, kata ganti nama khas'},
    {'Tag': 'SCONJ', 'Description': 'Subordinating conjunction'},
    {'Tag': 'SYM', 'Description': 'Symbol'},
    {'Tag': 'VERB', 'Description': 'Verb, kata kerja'},
    {'Tag': 'X', 'Description': 'Other'},
]

available_huggingface = {
    'mesolitica/pos-t5-tiny-standard-bahasa-cased': {
        'Size (MB)': 84.7,
        'PART': {'precision': 0.8938547486033519,
                 'recall': 0.9411764705882353,
                 'f1': 0.9169054441260744,
                 'number': 170},
        'CCONJ': {'precision': 0.9713905522288756,
                  'recall': 0.9785522788203753,
                  'f1': 0.974958263772955,
                  'number': 1492},
        'ADJ': {'precision': 0.9192897497982244,
                'recall': 0.88984375,
                'f1': 0.9043271139341008,
                'number': 1280},
        'ADP': {'precision': 0.9770908087220536,
                'recall': 0.9844271412680756,
                'f1': 0.9807452555755645,
                'number': 3596},
        'ADV': {'precision': 0.9478672985781991,
                'recall': 0.9523809523809523,
                'f1': 0.9501187648456056,
                'number': 1260},
        'VERB': {'precision': 0.9654357459379616,
                 'recall': 0.9662921348314607,
                 'f1': 0.9658637505541599,
                 'number': 3382},
        'DET': {'precision': 0.9603854389721628,
                'recall': 0.9542553191489361,
                'f1': 0.9573105656350054,
                'number': 940},
        'NOUN': {'precision': 0.8789933694996986,
                 'recall': 0.8976608187134503,
                 'f1': 0.8882290239074159,
                 'number': 6498},
        'PRON': {'precision': 0.9888991674375578,
                 'recall': 0.9861623616236163,
                 'f1': 0.9875288683602771,
                 'number': 1084},
        'PROPN': {'precision': 0.8842357164223751,
                  'recall': 0.8982072318444242,
                  'f1': 0.891166716912873,
                  'number': 6582},
        'NUM': {'precision': 0.9532391622016562,
                'recall': 0.9688118811881188,
                'f1': 0.9609624355511908,
                'number': 2020},
        'PUNCT': {'precision': 0.9991261796574624,
                  'recall': 0.9980796089385475,
                  'f1': 0.9986026200873362,
                  'number': 5728},
        'AUX': {'precision': 1.0,
                'recall': 0.9852941176470589,
                'f1': 0.9925925925925926,
                'number': 204},
        'SYM': {'precision': 0.8950617283950617,
                'recall': 0.90625,
                'f1': 0.9006211180124224,
                'number': 160},
        'X': {'precision': 0.4444444444444444,
              'recall': 0.5,
              'f1': 0.47058823529411764,
              'number': 16},
        'overall_precision': 0.9370964022140221,
        'overall_recall': 0.9446123445309775,
        'overall_f1': 0.9408393632416786,
        'overall_accuracy': 0.9579554043839759
    },
    'mesolitica/pos-t5-small-standard-bahasa-cased': {
        'Size (MB)': 141,
        'PART': {'precision': 0.950920245398773,
                 'recall': 0.9117647058823529,
                 'f1': 0.9309309309309309,
                 'number': 170},
        'SCONJ': {'precision': 0.9883481836874571,
                  'recall': 0.9664879356568364,
                  'f1': 0.9772958319213825,
                  'number': 1492},
        'ADJ': {'precision': 0.9257425742574258,
                'recall': 0.8765625,
                'f1': 0.9004815409309791,
                'number': 1280},
        'ADP': {'precision': 0.9854219231847491,
                'recall': 0.9774749721913237,
                'f1': 0.9814323607427056,
                'number': 3596},
        'ADV': {'precision': 0.9580306698950767,
                'recall': 0.942063492063492,
                'f1': 0.9499799919967987,
                'number': 1260},
        'VERB': {'precision': 0.9693969396939695,
                 'recall': 0.9553518628030752,
                 'f1': 0.9623231571109457,
                 'number': 3382},
        'DET': {'precision': 0.9666307857911733,
                'recall': 0.9553191489361702,
                'f1': 0.9609416800428037,
                'number': 940},
        'NOUN': {'precision': 0.892811906269791,
                 'recall': 0.8678054786088027,
                 'f1': 0.880131106602154,
                 'number': 6498},
        'PRON': {'precision': 0.9906803355079217,
                 'recall': 0.9806273062730627,
                 'f1': 0.9856281872971719,
                 'number': 1084},
        'PROPN': {'precision': 0.8682452062754212,
                  'recall': 0.9080826496505622,
                  'f1': 0.8877172137234517,
                  'number': 6582},
        'NUM': {'precision': 0.9799899949974987,
                'recall': 0.9698019801980198,
                'f1': 0.9748693704901717,
                'number': 2020},
        'PUNCT': {'precision': 0.9986033519553073,
                  'recall': 0.9986033519553073,
                  'f1': 0.9986033519553073,
                  'number': 5728},
        'AUX': {'precision': 0.9900990099009901,
                'recall': 0.9803921568627451,
                'f1': 0.9852216748768472,
                'number': 204},
        'SYM': {'precision': 0.9246575342465754,
                'recall': 0.84375,
                'f1': 0.8823529411764707,
                'number': 160},
        'X': {'precision': 1.0, 'recall': 0.25, 'f1': 0.4, 'number': 16},
        'overall_precision': 0.941408302679979,
        'overall_recall': 0.9370859002673486,
        'overall_f1': 0.939242128564355,
        'overall_accuracy': 0.955475245653817
    },
}


def _naive_POS_word(word):
    for key, vals in tatabahasa_dict.items():
        if word in vals:
            return (key, word)
    try:
        if len(re.findall(r'^(.*?)(%s)$' % ('|'.join(hujung[:1])), i)[0]) > 1:
            return ('KJ', word)
    except BaseException:
        pass
    try:
        if (
            len(re.findall(r'^(.*?)(%s)' % ('|'.join(permulaan[:-4])), word)[0])
            > 1
        ):
            return ('KJ', word)
    except Exception as e:
        pass
    if len(word) > 2:
        return ('KN', word)
    else:
        return ('', word)


def huggingface(
    model: str = 'mesolitica/pos-t5-small-standard-bahasa-cased',
    force_check: bool = True,
    **kwargs,
):
    """
    Load HuggingFace model to Part-of-Speech Recognition.

    Parameters
    ----------
    model: str, optional (default='mesolitica/pos-t5-small-standard-bahasa-cased')
        Check available models at `malaya.pos.available_huggingface`.
    force_check: bool, optional (default=True)
        Force check model one of malaya model.
        Set to False if you have your own huggingface model.

    Returns
    -------
    result: malaya.torch_model.huggingface.Tagging
    """

    return load(
        model=model,
        class_model=Tagging,
        available_huggingface=available_huggingface,
        force_check=force_check,
        path=__name__,
        **kwargs,
    )
