import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MALAYA_USE_HUGGINGFACE'] = 'true'

import sys
import malaya
import logging

logging.basicConfig(level=logging.DEBUG)


def test_wordvector():
    vocab_news, embedded_news = malaya.wordvector.load(model='news')
    word_vector_news = malaya.wordvector.WordVector(embedded_news, vocab_news)

    word = 'anwar'
    word_vector_news.n_closest(word=word, num_closest=8, metric='cosine')

    words = ['anwar', 'mahathir']
    word_vector_news.batch_n_closest(words, num_closest=8,
                                     return_similarity=False)

    word_vector_news.calculator('anwar + amerika + mahathir', num_closest=8, metric='cosine',
                                return_similarity=False)

    os.system('rm -f ~/.cache/huggingface/hub/*')
    del word_vector_news, vocab_news, embedded_news
