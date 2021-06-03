import re
import numpy as np
from malaya.text.bahasa.lapor import lapor as _lapor_words
from malaya.text.bahasa.news import news as _news_words
from malaya.text.function import split_into_sentences
from malaya.text.ngram import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances

split_words = ['SPPPPLIIIT>', 'SPPPPLIIIT']


def _rouge_clean(s):
    s = re.sub(r'[^a-zA-Z0-9 ]', '', s)
    return re.sub(r'[ ]+', ' ', s).strip().lower()


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i: i + n]))
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


def postprocessing_summarization(string, lapors=_lapor_words):
    for l in lapors:
        if l in string:
            string = re.sub(f'\\s*[,.]?\\s*{l}', ' ', string)

    string = re.sub(r'[ ]+', ' ', string).strip()
    return string


def find_lapor_and_remove(article, summary):
    lapor = []
    lowered = article.lower()
    finds = re.findall('\\w*lapor \\w*', summary)
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


def filter_news_sentence(summary):
    sents = split_into_sentences(summary)
    selected = []
    for s in sents:
        s_lower = s.lower()
        if all([n not in s_lower for n in _news_words]):
            selected.append(s)
    return ' '.join(selected)


def get_unique_sentences(summary, reject_similarity=0.85, **kwargs):

    sents = split_into_sentences(summary)
    bow = CountVectorizer(token_pattern='[A-Za-z]+').fit_transform(sents)
    coef = 1 - pairwise_distances(X=bow, Y=bow, metric='cosine')
    ids, selected = [], []
    for no in range(len(sents)):
        row = np.where(coef[no] >= reject_similarity)[0]
        if row[0] not in selected:
            ids.append(sents[row[0]])
            selected.extend(row)

    return ' '.join(ids)


def filter_rouge(article, summary, n=2, threshold=0.1, **kwargs):

    sents = split_into_sentences(summary)
    reference = _get_word_ngrams(n, [_rouge_clean(article).split()])
    results = []
    for s in sents:
        evaluated = _get_word_ngrams(n, [_rouge_clean(s).split()])
        score = cal_rouge(evaluated, reference)['p']
        if score >= threshold:
            results.append(s)
    return ' '.join(results)


def postprocess_summary(string, summary, **kwargs):
    summary = filter_rouge(string, summary, **kwargs)
    summary = postprocessing_summarization(summary)
    summary = find_lapor_and_remove(string, summary)
    summary = filter_news_sentence(summary)
    summary = get_unique_sentences(summary, **kwargs)
    for s in split_words:
        summary = summary.replace(s, ' ')
    return re.sub(r'[ ]+', ' ', summary).strip()
