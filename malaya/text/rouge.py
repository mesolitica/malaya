# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/rouge.py

import re
import numpy as np
from malaya.text.bahasa.lapor import lapor as _lapor_words
from malaya.text.bahasa.news import news as _news_words
from malaya.text.function import split_into_sentences, remove_empty_parenthesis
from malaya.text.ngram import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances

split_words = ['SPPPPLIIIT>', 'SPPPPLIIIT']


def _len_lcs(x, y):
    """Returns the length of the Longest Common Subsequence between two seqs.
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: sequence of words
      y: sequence of words
    Returns
      integer: Length of LCS between x and y
    """
    table = _lcs(x, y)
    n, m = len(x), len(y)
    return table[n, m]


def _lcs(x, y):
    """Computes the length of the LCS between two seqs.
    The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
    Args:
      x: collection of words
      y: collection of words
    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = {}
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def _f_lcs(llcs, m, n):
    """Computes the LCS-based F-measure score.
    Source: https://www.microsoft.com/en-us/research/publication/
    rouge-a-package-for-automatic-evaluation-of-summaries/
    Args:
      llcs: Length of LCS
      m: number of words in reference summary
      n: number of words in candidate summary
    Returns:
      Float. LCS-based F-measure score
    """
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + (beta**2)) * r_lcs * p_lcs
    denom = r_lcs + ((beta**2) * p_lcs)
    f_lcs = num / (denom + 1e-12)
    return f_lcs


def _rouge_clean(s):
    s = re.sub(r'[^a-zA-Z0-9 ]', '', s)
    return re.sub(r'[ ]+', ' ', s).strip().lower()


def rouge_l(eval_sentence, ref_sentence):
    """Computes ROUGE-L (sentence level) of two collections of sentences.
    Source: https://www.microsoft.com/en-us/research/publication/
    rouge-a-package-for-automatic-evaluation-of-summaries/
    Calculated according to:
    R_lcs = LCS(X,Y)/m
    P_lcs = LCS(X,Y)/n
    F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
    where:
    X = reference summary
    Y = Candidate summary
    m = length of reference summary
    n = length of candidate summary
    Args:
      eval_sentences: The sentences that have been picked by the summarizer
      ref_sentences: The sentences from the reference set
    Returns:
      A float: F_lcs
    """

    m = len(ref_sentence)
    n = len(eval_sentence)
    lcs = _len_lcs(eval_sentence, ref_sentence)
    return _f_lcs(lcs, m, n)


def rouge_n(eval_sentence, ref_sentence, n=2):

    eval_ngrams = _get_ngrams(n, eval_sentence)
    ref_ngrams = _get_ngrams(n, ref_sentence)
    ref_count = len(ref_ngrams)
    eval_count = len(eval_ngrams)

    overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if eval_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / eval_count

    if ref_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / ref_count

    return 2.0 * ((precision * recall) / (precision + recall + 1e-8))


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


def find_kata_encik(string, substring='kata Encik', **kwargs):
    finds = [(m.start(0), m.end(0)) for m in re.finditer(f'\\w*{substring}\\w*', string)]
    if len(finds):
        string = re.sub(r'[ ]+', ' ', string[:finds[0][0]]).strip()
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
    """
    Parameters
    ----------

    n: int, optional (default=2)
        N size of rouge to filter
    threshold: float, optional (default=0.1)
        minimum threshold for N rouge score to select a sentence.
    reject_similarity: float, optional (default=0.85)
        reject similar sentences while maintain position.
    min_length_inside: int, optional, (default=2)
        minimum length inside parenthesis to not reject.
    """

    summary = filter_rouge(string, summary, **kwargs)
    summary = postprocessing_summarization(summary)
    summary = find_lapor_and_remove(string, summary)
    summary = filter_news_sentence(summary)
    summary = get_unique_sentences(summary, **kwargs)
    summary = remove_empty_parenthesis(summary, **kwargs)
    for s in split_words:
        summary = summary.replace(s, ' ')
    return re.sub(r'[ ]+', ' ', summary).strip()
