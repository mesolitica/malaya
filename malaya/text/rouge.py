import re
from malaya.text.bahasa.lapor import lapor as _lapor_words
from malaya.text.function import split_into_sentences
from malaya.generator import ngrams


def _rouge_clean(s):
    s = re.sub(r'[^a-zA-Z0-9 ]', '', s)
    return re.sub(r'[ ]+', ' ', s).strip().lower()


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i : i + n]))
    return ngram_set


def _get_word_ngrams(n, sentences):
    assert len(sentences) > 0
    assert n > 0

    words = sum(sentences, [])
    return _get_ngrams(n, words)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {'f': f1_score, 'p': precision, 'r': recall}


def filter_rouge(article, summary, n = 2, threshold = 0.1, **kwargs):
    sents = split_into_sentences(summary)
    reference = _get_word_ngrams(n, [_rouge_clean(article).split()])
    results = []
    for s in sents:
        evaluated = _get_word_ngrams(n, [_rouge_clean(s).split()])
        score = cal_rouge(evaluated, reference)['p']
        if score >= threshold:
            results.append(s)
    return ' '.join(results)


def postprocessing_summarization(string, lapors = _lapor_words):
    for l in lapors:
        if l in string:
            string = re.sub(f'\s*[,.]?\s*{l}', ' ', string)

    string = re.sub(r'[ ]+', ' ', string).strip()
    return string


def find_lapor_and_remove(article, summary):
    lapor = []
    lowered = article.lower()
    finds = re.findall('\w*lapor \w*', summary)
    for f in finds:
        start = summary.find(f)
        end = summary.find('.', start)
        s = summary[start:end].split(',')
        s = s[0].split(';')
        s = s[0].split(':')
        s = s[0].split('-')
        if len(s[0].split()) < 8:
            a = s[0].replace('lapor ', '').lower().split()
            ngram = list(ngrams(lowered.split(), len(a)))
            if a not in ngram:
                lapor.append(s[0])

    summary = postprocessing_summarization(summary, lapor)
    return summary
