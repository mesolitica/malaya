import numpy as np
import itertools
from malaya.text.function import (
    summary_textcleaning,
    split_into_sentences,
    simple_textcleaning,
    STOPWORDS,
)
from malaya.cluster import cluster_words
from malaya.graph.pagerank import pagerank
from sklearn.metrics.pairwise import cosine_similarity
from herpetologist import check_type
from scipy.sparse import vstack
from typing import List, Tuple


def create_ngram(string, ngram=10):
    splitted = string.split()
    ngram_list = []
    for i in range(0, len(splitted)):
        lower_bound = i - ngram if i - ngram > 0 else 0
        upper_bound = (
            i + ngram if i + ngram < len(splitted) else len(splitted) - 1
        )
        new_word = splitted[lower_bound:upper_bound]
        new_string = ' '.join(new_word)
        if new_string == '':
            new_string = ' '
        ngram_list.append(new_string)

    return ngram_list, splitted


def corpus_checker(corpus):
    if not isinstance(corpus, list) and not isinstance(corpus, str):
        raise ValueError('corpus must be a list or a string')
    if isinstance(corpus, list):
        if not isinstance(corpus[0], str):
            raise ValueError('corpus must be list of strings')
    if isinstance(corpus, str):
        corpus = split_into_sentences(corpus, minimum_length=20)
    else:
        corpus = '. '.join(corpus)
        corpus = split_into_sentences(corpus, minimum_length=20)
    return corpus


class SKLearn:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def _vectorize_word(
        self, corpus, isi_penting, window_size, important_words=10, **kwargs
    ):
        corpus = corpus_checker(corpus)
        splitted_fullstop = [summary_textcleaning(i) for i in corpus]
        original_strings = [i[0] for i in splitted_fullstop]
        cleaned_strings = [i[1] for i in splitted_fullstop]
        ngram_list, splitted = create_ngram(
            ' '.join(cleaned_strings), ngram=window_size
        )
        splitted = ' '.join(original_strings).split()
        if isi_penting:
            isi_penting = [summary_textcleaning(isi_penting)[1]]
        else:
            isi_penting = [' '.join(cleaned_strings)]
        t = ngram_list + isi_penting
        self.vectorizer.fit(t)
        freq = self.vectorizer.transform(ngram_list)
        freq_isi_penting = self.vectorizer.transform(isi_penting)
        if important_words > 0:
            if hasattr(self.vectorizer, 'idf_'):
                indices = np.argsort(self.vectorizer.idf_)[::-1]
            else:
                indices = np.argsort(np.asarray(freq.sum(axis=0))[0])[::-1]
            features = self.vectorizer.get_feature_names()
            top_words = [features[i] for i in indices[:important_words]]
        else:
            top_words = []
        t = vstack([freq, freq_isi_penting])
        self.model.fit(t)
        vectors = self.model.transform(freq)
        vectors_isi_penting = self.model.transform(freq_isi_penting)
        similar_isi_penting = cosine_similarity(vectors, vectors_isi_penting)
        scores = similar_isi_penting[:, 0]
        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(splitted)), reverse=True
        )
        return (splitted, ranked_sentences, top_words, cluster_words(top_words))

    def _vectorize_sentence(
        self, corpus, isi_penting, important_words=10, retry=5, **kwargs
    ):
        corpus = corpus_checker(corpus)
        splitted_fullstop = [summary_textcleaning(i) for i in corpus]
        original_strings = [i[0] for i in splitted_fullstop]
        cleaned_strings = [i[1] for i in splitted_fullstop]
        if isi_penting:
            isi_penting = [summary_textcleaning(isi_penting)[1]]
            t = cleaned_strings + isi_penting
        else:
            t = cleaned_strings
        self.vectorizer.fit(t)
        freq = self.vectorizer.transform(cleaned_strings)
        if isi_penting:
            freq_isi_penting = self.vectorizer.transform(isi_penting)
        if important_words > 0:
            if hasattr(self.vectorizer, 'idf_'):
                indices = np.argsort(self.vectorizer.idf_)[::-1]
            else:
                indices = np.argsort(np.asarray(freq.sum(axis=0))[0])[::-1]
            features = self.vectorizer.get_feature_names()
            top_words = [features[i] for i in indices[:important_words]]
        else:
            top_words = []
        if isi_penting:
            t = vstack([freq, freq_isi_penting])
        else:
            t = freq
        self.model.fit(t)
        vectors = self.model.transform(freq)
        if isi_penting:
            vectors_isi_penting = self.model.transform(freq_isi_penting)
        similar = cosine_similarity(vectors, vectors)
        if isi_penting:
            similar_isi_penting = cosine_similarity(
                vectors, vectors_isi_penting
            )
            similar = similar * similar_isi_penting
        else:
            similar[similar >= 0.99] = 0
        scores = pagerank(similar + 1e-6, retry, **kwargs)
        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(original_strings)),
            reverse=True,
        )
        return (
            original_strings,
            ranked_sentences,
            top_words,
            cluster_words(top_words),
        )

    @check_type
    def word_level(
        self,
        corpus,
        isi_penting: str = None,
        window_size: int = 10,
        important_words: int = 10,
        **kwargs
    ):
        """
        Summarize list of strings / string on word level.

        Parameters
        ----------
        corpus: str / List[str]
        isi_penting: str, optional (default=None)
            if not None, will put priority based on `isi_penting`.
        window_size: int, (default=10)
            window size for each word.
        important_words: int, (default=10)
            number of important words.

        Returns
        -------
        dict: {'top-words', 'cluster-top-words', 'score'}
        """
        splitted, ranked_sentences, top_words, cluster_top_words = self._vectorize_word(
            corpus,
            isi_penting=isi_penting,
            window_size=window_size,
            important_words=important_words,
            **kwargs
        )
        for score, s, rank in ranked_sentences:
            s = (splitted[rank], score)
            splitted[rank] = s
        return {
            'top-words': top_words,
            'cluster-top-words': cluster_top_words,
            'score': splitted,
        }

    @check_type
    def sentence_level(
        self,
        corpus,
        isi_penting: str = None,
        top_k: int = 3,
        important_words: int = 10,
        **kwargs
    ):
        """
        Summarize list of strings / string on sentence level.

        Parameters
        ----------
        corpus: str / List[str]
        isi_penting: str, optional (default=None)
            if not None, will put priority based on `isi_penting`.
        top_k: int, (default=3)
            number of summarized strings.
        important_words: int, (default=10)
            number of important words.

        Returns
        -------
        dict: {'summary', 'top-words', 'cluster-top-words', 'score'}
        """
        original_strings, ranked_sentences, top_words, cluster_top_words = self._vectorize_sentence(
            corpus,
            isi_penting=isi_penting,
            important_words=important_words,
            **kwargs
        )
        for score, s, rank in ranked_sentences:
            s = original_strings[rank].split()
            s = [(w, score) for w in s]
            original_strings[rank] = s
        merged = list(itertools.chain(*original_strings))
        summary = [r[1] for r in ranked_sentences[:top_k]]
        return {
            'summary': ' '.join(summary),
            'top-words': top_words,
            'cluster-top-words': cluster_top_words,
            'score': merged,
        }


class Doc2Vec:
    def __init__(self, wordvector):
        self.wordvector = wordvector

    def _vectorize_word(
        self,
        corpus,
        isi_penting,
        window_size,
        aggregation=np.mean,
        soft=False,
        **kwargs
    ):
        corpus = corpus_checker(corpus)
        splitted_fullstop = [summary_textcleaning(i) for i in corpus]
        original_strings = [i[0] for i in splitted_fullstop]
        cleaned_strings = [i[1] for i in splitted_fullstop]
        ngram_list, splitted = create_ngram(
            ' '.join(cleaned_strings), ngram=window_size
        )
        splitted = ' '.join(original_strings).split()
        if isi_penting:
            isi_penting = summary_textcleaning(isi_penting)[1]
        else:
            isi_penting = ' '.join(cleaned_strings)
        vectors = []
        for string in ngram_list:
            inside = []
            for token in string.split():
                if token in self.wordvector._dictionary:
                    v = self.wordvector.get_vector_by_name(token)
                else:
                    if not soft:
                        v = np.zeros((self.wordvector._embed_matrix.shape[1]))
                    else:
                        arr = np.array(
                            [
                                self.wordvector._jarowinkler.similarity(
                                    token, k
                                )
                                for k in self.wordvector.words
                            ]
                        )
                        idx = (-arr).argsort()[0]
                        v = self.wordvector.get_vector_by_name(
                            self.wordvector.words[idx]
                        )
                inside.append(v)
            vectors.append(aggregation(inside, axis=0))
        vectors = np.array(vectors)

        cleaned_isi_penting = isi_penting
        vectors_isi_penting = []
        for token in cleaned_isi_penting.split():
            if token in self.wordvector._dictionary:
                vectors_isi_penting.append(
                    self.wordvector.get_vector_by_name(token)
                )
            else:
                if not soft:
                    vectors_isi_penting.append(
                        np.zeros((self.wordvector._embed_matrix.shape[1]))
                    )
                else:
                    arr = np.array(
                        [
                            self.wordvector._jarowinkler.similarity(token, k)
                            for k in self.wordvector.words
                        ]
                    )
                    idx = (-arr).argsort()[0]
                    vectors_isi_penting.append(
                        self.wordvector.get_vector_by_name(
                            self.wordvector.words[idx]
                        )
                    )
        vectors_isi_penting = aggregation(vectors_isi_penting, axis=0)
        vectors_isi_penting = np.expand_dims(vectors_isi_penting, 0)
        similar_isi_penting = cosine_similarity(vectors, vectors_isi_penting)
        scores = similar_isi_penting[:, 0]
        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(splitted)), reverse=True
        )
        return (splitted, ranked_sentences)

    def _vectorize_sentence(
        self,
        corpus,
        isi_penting,
        aggregation=np.mean,
        soft=False,
        retry=5,
        **kwargs
    ):

        corpus = corpus_checker(corpus)
        splitted_fullstop = [summary_textcleaning(i) for i in corpus]
        original_strings = [i[0] for i in splitted_fullstop]
        cleaned_strings = [i[1] for i in splitted_fullstop]
        vectors = []
        for string in cleaned_strings:
            inside = []
            for token in string.split():
                if token in self.wordvector._dictionary:
                    v = self.wordvector.get_vector_by_name(token)
                else:
                    if not soft:
                        v = np.zeros((self.wordvector._embed_matrix.shape[1]))

                    else:
                        arr = np.array(
                            [
                                self.wordvector._jarowinkler.similarity(
                                    token, k
                                )
                                for k in self.wordvector.words
                            ]
                        )
                        idx = (-arr).argsort()[0]
                        v = self.wordvector.get_vector_by_name(
                            self.wordvector.words[idx]
                        )

                inside.append(v)
            vectors.append(aggregation(inside, axis=0))
        vectors = np.array(vectors)

        if isi_penting:
            cleaned_isi_penting = summary_textcleaning(isi_penting)[1]
            vectors_isi_penting = []
            for token in cleaned_isi_penting.split():
                if token in self.wordvector._dictionary:
                    v = self.wordvector.get_vector_by_name(token)
                else:
                    if not soft:
                        v = np.zeros((self.wordvector._embed_matrix.shape[1]))
                    else:
                        arr = np.array(
                            [
                                self.wordvector._jarowinkler.similarity(
                                    token, k
                                )
                                for k in self.wordvector.words
                            ]
                        )
                        idx = (-arr).argsort()[0]
                        v = self.wordvector.get_vector_by_name(
                            self.wordvector.words[idx]
                        )
                vectors_isi_penting.append(v)
            vectors_isi_penting = aggregation(vectors_isi_penting, axis=0)
            vectors_isi_penting = np.expand_dims(vectors_isi_penting, 0)

        similar = cosine_similarity(vectors, vectors)
        if isi_penting:
            similar_isi_penting = cosine_similarity(
                vectors, vectors_isi_penting
            )
            similar = similar * similar_isi_penting
        else:
            similar[similar >= 0.99] = 0
        scores = pagerank(similar + 1e-6, retry, **kwargs)
        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(original_strings)),
            reverse=True,
        )
        return (original_strings, ranked_sentences)

    def word_level(
        self,
        corpus,
        isi_penting: str = None,
        window_size: int = 10,
        aggregation=np.mean,
        soft: bool = False,
        **kwargs
    ):
        """
        Summarize list of strings / string on sentence level.

        Parameters
        ----------
        corpus: str / List[str]
        isi_penting: str, optional (default=None)
            if not None, will put priority based on `isi_penting`.
        window_size: int, (default=10)
            window size for each word.
        aggregation: Callable, optional (default=numpy.mean)
            Aggregation method for Doc2Vec.
        soft: bool, optional (default=False)
            soft: bool, (default=True)
            if True, a word not in the dictionary will be replaced with nearest JaroWinkler ratio.
            if False, it will returned embedding full with zeros.

        Returns
        -------
        dict: {'score'}
        """
        splitted, ranked_sentences = self._vectorize_word(
            corpus,
            isi_penting=isi_penting,
            window_size=window_size,
            aggregation=np.mean,
            soft=soft,
        )
        for score, s, rank in ranked_sentences:
            s = (splitted[rank], score)
            splitted[rank] = s
        return {'score': splitted}

    def sentence_level(
        self,
        corpus,
        isi_penting: str = None,
        top_k: int = 3,
        aggregation=np.mean,
        soft: bool = False,
        **kwargs
    ):
        """
        Summarize list of strings / string on sentence level.

        Parameters
        ----------
        corpus: str / List[str]
        isi_penting: str, optional (default=None)
            if not None, will put priority based on `isi_penting`.
        top_k: int, (default=3)
            number of summarized strings.
        aggregation: Callable, optional (default=numpy.mean)
            Aggregation method for Doc2Vec.
        soft: bool, optional (default=False)
            soft: bool, (default=True)
            if True, a word not in the dictionary will be replaced with nearest JaroWinkler ratio.
            if False, it will returned embedding full with zeros.

        Returns
        -------
        dict: {'summary', 'score'}
        """
        original_strings, ranked_sentences = self._vectorize_sentence(
            corpus,
            isi_penting=isi_penting,
            aggregation=aggregation,
            soft=soft,
            **kwargs
        )
        for score, s, rank in ranked_sentences:
            s = original_strings[rank].split()
            s = [(w, score) for w in s]
            original_strings[rank] = s
        merged = list(itertools.chain(*original_strings))
        summary = [r[1] for r in ranked_sentences[:top_k]]
        return {'summary': ' '.join(summary), 'score': merged}


class Encoder:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def _batching(self, l, batch_size=10):
        vs = []
        for i in range(0, len(l), batch_size):
            index = min(i + batch_size, len(l))
            batch_x = l[i:index]
            vectors = self.vectorizer.vectorize(batch_x)
            vs.append(vectors)
        return np.concatenate(vs, axis=0)

    def _vectorize_word(
        self,
        corpus,
        isi_penting,
        window_size=10,
        important_words=10,
        batch_size=10,
        **kwargs
    ):
        corpus = corpus_checker(corpus)
        splitted_fullstop = [summary_textcleaning(i) for i in corpus]
        original_strings = [i[0] for i in splitted_fullstop]
        cleaned_strings = [i[1] for i in splitted_fullstop]
        ngram_list, splitted = create_ngram(
            ' '.join(cleaned_strings), ngram=window_size
        )
        splitted = ' '.join(original_strings).split()
        if isi_penting:
            isi_penting = [isi_penting]
        else:
            isi_penting = original_strings

        vectors = self._batching(ngram_list, batch_size=batch_size)
        vectors_isi_penting = self._batching(
            isi_penting, batch_size=batch_size
        )

        if 'DeepSkipThought' in str(self.vectorizer):
            top_words = []
        else:
            if hasattr(self.vectorizer, 'attention') and important_words > 0:
                attentions = self.vectorizer.attention(corpus, **kwargs)
                flatten = list(itertools.chain(*attentions))
                r = {}
                for f in flatten:
                    c = simple_textcleaning(f[0])
                    if c in STOPWORDS:
                        continue
                    if c not in r:
                        r[c] = f[1]
                    else:
                        r[c] += f[1]
                top_words = sorted(r, key=r.get, reverse=True)[
                    :important_words
                ]
            else:
                top_words = []

        vectors_isi_penting = np.mean(vectors_isi_penting, axis=0)
        vectors_isi_penting = np.expand_dims(vectors_isi_penting, axis=0)
        similar_isi_penting = cosine_similarity(vectors, vectors_isi_penting)
        scores = similar_isi_penting[:, 0]
        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(splitted)), reverse=True
        )
        return (splitted, ranked_sentences, top_words, cluster_words(top_words))

    def _vectorize_sentence(
        self,
        corpus,
        isi_penting,
        important_words=10,
        batch_size=10,
        retry=5,
        **kwargs
    ):
        corpus = corpus_checker(corpus)
        splitted_fullstop = [summary_textcleaning(i) for i in corpus]
        original_strings = [i[0] for i in splitted_fullstop]
        cleaned_strings = [i[1] for i in splitted_fullstop]

        vectors = self._batching(cleaned_strings, batch_size=batch_size)
        if isi_penting:
            vectors_isi_penting = self._batching(
                [isi_penting], batch_size=batch_size
            )

        if 'DeepSkipThought' in str(self.vectorizer):
            top_words = []
        else:
            if hasattr(self.vectorizer, 'attention'):
                attentions = self.vectorizer.attention(corpus, **kwargs)
                flatten = list(itertools.chain(*attentions))
                r = {}
                for f in flatten:
                    c = simple_textcleaning(f[0])
                    if c in STOPWORDS:
                        continue
                    if c not in r:
                        r[c] = f[1]
                    else:
                        r[c] += f[1]
                top_words = sorted(r, key=r.get, reverse=True)[
                    :important_words
                ]
            else:
                top_words = []

        similar = cosine_similarity(vectors, vectors)
        if isi_penting:
            similar_isi_penting = cosine_similarity(
                vectors, vectors_isi_penting
            )
            similar = similar * similar_isi_penting
        else:
            similar[similar >= 0.99] = 0
        scores = pagerank(similar + 1e-6, retry, **kwargs)
        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(original_strings)),
            reverse=True,
        )
        return (
            original_strings,
            ranked_sentences,
            top_words,
            cluster_words(top_words),
        )

    def word_level(
        self,
        corpus,
        isi_penting: str = None,
        window_size: int = 10,
        important_words: int = 10,
        batch_size: int = 16,
        **kwargs
    ):
        """
        Summarize list of strings / string on word level.

        Parameters
        ----------
        corpus: str / List[str]
        isi_penting: str, optional (default=None)
            if not None, will put priority based on `isi_penting`.
        window_size: int, (default=10)
            window size for each word.
        important_words: int, (default=10)
            number of important words.
        batch_size: int, (default=16)
            for each feed-forward, we only feed N size of texts for each batch.
            This to prevent OOM.

        Returns
        -------
        dict: {'summary', 'top-words', 'cluster-top-words', 'score'}
        """
        splitted, ranked_sentences, top_words, cluster_top_words = self._vectorize_word(
            corpus,
            isi_penting=isi_penting,
            window_size=window_size,
            important_words=important_words,
            batch_size=batch_size,
            **kwargs
        )
        for score, s, rank in ranked_sentences:
            s = (splitted[rank], score)
            splitted[rank] = s
        return {
            'top-words': top_words,
            'cluster-top-words': cluster_top_words,
            'score': splitted,
        }

    def sentence_level(
        self,
        corpus,
        isi_penting: str = None,
        top_k: int = 3,
        important_words: int = 10,
        batch_size: int = 16,
        **kwargs
    ):
        """
        Summarize list of strings / string on sentence level.

        Parameters
        ----------
        corpus: str / List[str]
        isi_penting: str, optional (default=None)
            if not None, will put priority based on `isi_penting`.
        top_k: int, (default=3)
            number of summarized strings.
        important_words: int, (default=10)
            number of important words.
        batch_size: int, (default=16)
            for each feed-forward, we only feed N size of texts for each batch.
            This to prevent OOM.

        Returns
        -------
        dict: {'summary', 'top-words', 'cluster-top-words', 'score'}
        """
        original_strings, ranked_sentences, top_words, cluster_top_words = self._vectorize_sentence(
            corpus,
            isi_penting=isi_penting,
            important_words=important_words,
            batch_size=batch_size,
            **kwargs
        )
        for score, s, rank in ranked_sentences:
            s = original_strings[rank].split()
            s = [(w, score) for w in s]
            original_strings[rank] = s
        merged = list(itertools.chain(*original_strings))
        summary = [r[1] for r in ranked_sentences[:top_k]]
        return {
            'summary': ' '.join(summary),
            'top-words': top_words,
            'cluster-top-words': cluster_top_words,
            'score': merged,
        }
