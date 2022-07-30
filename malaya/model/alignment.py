"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

https://github.com/robertostling/eflomal/blob/master/LICENSE

Copy from https://github.com/robertostling/eflomal/blob/master/align.py optimized using defaultdict,

left = ['Terminal 1 KKIA dilengkapi kemudahan 64 kaunter daftar masuk, 12 aero bridge selain mampu menampung 3,200 penumpang dalam satu masa.']
right = ['Terminal 1 KKIA is equipped with 64 check-in counters, 12 aero bridges and can accommodate 3,200 passengers at a time.']
eflomal_model.align(left, right) originally ~4 seconds, now ~140 ms.
"""

import numpy as np
from malaya.text.bpe import padding_sequence
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import List
import tensorflow as tf
import itertools
import logging

logger = logging.getLogger(__name__)


def read_text(text, lowercase=True):
    index = {}
    sents = []
    for line in text:
        if lowercase:
            tokens = line.lower().split()
        else:
            tokens = line.split()
        n = len(tokens)
        sent = np.empty(n, dtype=np.uint32)

        for i in range(n):
            token = tokens[i]
            idx = index.get(token, -1)
            if idx == -1:
                idx = len(index)
                index[token] = idx
            sent[i] = idx

        sents.append(sent)

    return sents, index


class Eflomal:
    def __init__(self, priors_filename, preprocessing_func=None, **kwargs):

        try:
            from eflomal import read_text, write_text, align
        except BaseException:
            raise ModuleNotFoundError(
                'eflomal not installed. Please install it from https://github.com/robertostling/eflomal for Linux / Windows or https://github.com/huseinzol05/maceflomal for Mac and try again.'
            )

        self._read_text = read_text
        self._write_text = write_text
        self._align = align
        self._priors_filename = priors_filename
        if preprocessing_func is None:
            self._preprocessing_func = lambda x: x
        else:
            self._preprocessing_func = preprocessing_func

        self._process_priors()

    def __del__(self):
        try:
            self._priors_list_dict.clear()
            self._ferf_priors_dict.clear()
            self._ferr_priors_dict.clear()
            self._hmmf_priors.clear()
            self._hmmr_priors.clear()
        except:
            pass

    def _process_priors(self):

        self._priors_list_dict = defaultdict(list)
        self._ferf_priors_dict = defaultdict(list)
        self._ferr_priors_dict = defaultdict(list)
        self._hmmf_priors = {}
        self._hmmr_priors = {}

        logger.debug('Caching Eflomal priors, will take some time.')

        with open(self._priors_filename, 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                fields = line.rstrip('\n').split('\t')
                try:
                    alpha = float(fields[-1])
                except ValueError:
                    raise ValueError(
                        'ERROR: priors file %s line %d contains alpha value of "%s" which is not numeric' %
                        (self_.priors_filename, i+1, fields[2]))

                if fields[0] == 'LEX' and len(fields) == 4:
                    k = f'{self._preprocessing_func(fields[1].lower())}-{self._preprocessing_func(fields[2].lower())}'
                    self._priors_list_dict[k].append(alpha)
                elif fields[0] == 'HMMF' and len(fields) == 3:
                    self._hmmf_priors[int(fields[1])] = alpha
                elif fields[0] == 'HMMR' and len(fields) == 3:
                    self._hmmr_priors[int(fields[1])] = alpha
                elif fields[0] == 'FERF' and len(fields) == 4:
                    self._ferf_priors_dict[self._preprocessing_func(fields[1].lower())].append((int(fields[2]), alpha))
                elif fields[0] == 'FERR' and len(fields) == 4:
                    self._ferr_priors_dict[self._preprocessing_func(fields[1].lower())].append((int(fields[2]), alpha))
                else:
                    raise ValueError('ERROR: priors file %s line %d is invalid ' % (self._priors_filename, i+1))

                i += 1

            self._total_lines = i

    def align(
        self,
        source: List[str],
        target: List[str],
        model: int = 3,
        score_model: int = 0,
        n_samplers: int = 3,
        length: float = 1.0,
        null_prior: float = 0.2,
        lowercase: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        """
        align text using eflomal, https://github.com/robertostling/eflomal/blob/master/align.py

        Parameters
        ----------
        source: List[str]
        target: List[str]
        model: int, optional (default=3)
            Model (1 = IBM1, 2 = IBM1+HMM, 3 = IBM1+HMM+fertility).
        score_model: int, optional (default=0)
            (1 = IBM1, 2 = IBM1+HMM, 3 = IBM1+HMM+fertility).
        n_samplers: int, optional (default=3)
            Number of independent samplers to run.
        length: float, optional (default=1.0)
            Relative number of sampling iterations.
        null_prior: float, optional (default=0.2)
            Prior probability of NULL alignment.
        lowercase: bool, optional (default=True)
            lowercase during searching priors.
        debug: bool, optional (default=False)
            debug `eflomal` binary.

        Returns
        -------
        result: Dict[List[List[Tuple]]]
        """
        if len(source) != len(target):
            raise ValueError('length source must be same as length of target')

        src_sents, src_index = read_text(source, lowercase=lowercase)
        n_src_sents = len(src_sents)
        src_voc_size = len(src_index)
        srcf = NamedTemporaryFile('wb')
        self._write_text(srcf, tuple(src_sents), src_voc_size)
        src_sents = None
        src_text = None

        trg_sents, trg_index = read_text(target, lowercase=lowercase)
        trg_voc_size = len(trg_index)
        n_trg_sents = len(trg_sents)
        trgf = NamedTemporaryFile('wb')
        self._write_text(trgf, tuple(trg_sents), trg_voc_size)
        trg_sents = None
        trg_text = None

        def get_src_index(src_word):
            src_word = src_word.lower()
            e = src_index.get(src_word)
            if e is not None:
                e = e + 1
            return e

        def get_trg_index(trg_word):
            trg_word = trg_word.lower()
            f = trg_index.get(trg_word)
            if f is not None:
                f = f + 1
            return f

        priors_indexed = {}
        for k in src_index:
            for v in trg_index:
                e = get_src_index(k)
                f = get_trg_index(v)

                key = f'{self._preprocessing_func(k.lower())}-{self._preprocessing_func(v.lower())}'
                if key in self._priors_list_dict:
                    for n in range(len(self._priors_list_dict[key])):
                        priors_indexed[(e, f)] = priors_indexed.get((e, f), 0.0) + self._priors_list_dict[key][n]

        ferf_indexed = {}
        for k in src_index:
            e = get_src_index(k)

            key = self._preprocessing_func(k.lower())
            if key in self._ferf_priors_dict:
                for n in range(len(self._ferf_priors_dict[key])):
                    fert = self._ferf_priors_dict[key][n][0]
                    alpha = self._ferf_priors_dict[key][n][1]
                    ferf_indexed[(e, fert)] = ferf_indexed.get((e, fert), 0.0) + alpha

        ferr_indexed = {}
        for k in trg_index:
            f = get_trg_index(k)
            key = self._preprocessing_func(k.lower())
            if key in self._ferr_priors_dict:
                for n in range(len(self._ferr_priors_dict[key])):
                    fert = self._ferr_priors_dict[key][n][0]
                    alpha = self._ferr_priors_dict[key][n][1]
                    ferr_indexed[(f, fert)] = \
                        ferr_indexed.get((f, fert), 0.0) + alpha

        priorsf = NamedTemporaryFile('w', encoding='utf-8')
        print('%d %d %d %d %d %d %d' % (
            len(src_index)+1, len(trg_index)+1, len(priors_indexed),
            len(self._hmmf_priors), len(self._hmmr_priors),
            len(ferf_indexed), len(ferr_indexed)),
            file=priorsf)

        for (e, f), alpha in sorted(priors_indexed.items()):
            print('%d %d %g' % (e, f, alpha), file=priorsf)

        for jump, alpha in sorted(self._hmmf_priors.items()):
            print('%d %g' % (jump, alpha), file=priorsf)

        for jump, alpha in sorted(self._hmmr_priors.items()):
            print('%d %g' % (jump, alpha), file=priorsf)

        for (e, fert), alpha in sorted(ferf_indexed.items()):
            print('%d %d %g' % (e, fert, alpha), file=priorsf)

        for (f, fert), alpha in sorted(ferr_indexed.items()):
            print('%d %d %g' % (f, fert, alpha), file=priorsf)

        priorsf.flush()

        trg_index = None
        src_index = None
        iters = None

        links_filename_fwd = NamedTemporaryFile('w')
        links_filename_rev = NamedTemporaryFile('w')

        self._align(srcf.name, trgf.name,
                    links_filename_fwd=links_filename_fwd.name,
                    links_filename_rev=links_filename_rev.name,
                    priors_filename=priorsf.name,
                    model=model,
                    score_model=score_model,
                    n_iterations=iters,
                    n_samplers=n_samplers,
                    quiet=not debug,
                    rel_iterations=length,
                    null_prior=null_prior,
                    use_gdb=debug)

        srcf.close()
        trgf.close()
        priorsf.close()

        links_filename_fwd.flush()
        links_filename_rev.flush()
        with open(links_filename_fwd.name) as fopen:
            fwd = fopen.read().strip()
        with open(links_filename_rev.name) as fopen:
            rev = fopen.read().strip()

        links_filename_fwd.close()
        links_filename_rev.close()

        fwd = fwd.split('\n')
        fwd_results = []
        for row in fwd:
            fwd_results_ = []
            for a in row.split():
                splitted = a.split('-')
                fwd_results_.append((int(splitted[0]), int(splitted[1])))
            fwd_results.append(fwd_results_)

        rev = rev.split('\n')
        rev_results = []
        for row in rev:
            rev_results_ = []
            for a in row.split():
                splitted = a.split('-')
                rev_results_.append((int(splitted[0]), int(splitted[1])))
            rev_results.append(rev_results_)

        return {'forward': fwd_results, 'reverse': rev_results}


class HuggingFace:
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def align(
        self,
        source: List[str],
        target: List[str],
        align_layer: int = 8,
        threshold: float = 1e-3,
    ):
        """
        align text using softmax output layers.

        Parameters
        ----------
        source: List[str]
        target: List[str]
        align_layer: int, optional (default=3)
            transformer layer-k to choose for embedding output.
        threshold: float, optional (default=1e-3)
            minimum probability to assume as alignment.

        Returns
        -------
        result: List[List[Tuple]]
        """

        if len(source) != len(target):
            raise ValueError('length source must be same as length of target')

        if align_layer >= self._model.config.num_hidden_layers:
            raise ValueError(f'`align_layer` must be < {self._model.config.num_hidden_layers}')

        input_ids_src, token_type_ids_src, attention_mask_src = [], [], []
        input_ids_tgt, token_type_ids_tgt, attention_mask_tgt = [], [], []
        sub2word_map_srcs, sub2word_map_tgts = [], []
        for i in range(len(source)):
            sent_src, sent_tgt = source[i].strip().split(), target[i].strip().split()
            token_src, token_tgt = [self._tokenizer.tokenize(word) for word in sent_src], [
                self._tokenizer.tokenize(word) for word in sent_tgt]
            wid_src, wid_tgt = [self._tokenizer.convert_tokens_to_ids(x) for x in token_src], [
                self._tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
            ids_src = self._tokenizer.prepare_for_model(list(itertools.chain(
                *wid_src)), return_tensors='np', model_max_length=self._tokenizer.model_max_length, truncation=True)
            input_ids_src.append(ids_src['input_ids'].tolist())
            token_type_ids_src.append(ids_src['token_type_ids'].tolist())
            attention_mask_src.append(ids_src['attention_mask'].tolist())

            ids_src = self._tokenizer.prepare_for_model(list(itertools.chain(
                *wid_tgt)), return_tensors='np', model_max_length=self._tokenizer.model_max_length, truncation=True)
            input_ids_tgt.append(ids_src['input_ids'].tolist())
            token_type_ids_tgt.append(ids_src['token_type_ids'].tolist())
            attention_mask_tgt.append(ids_src['attention_mask'].tolist())

            sub2word_map_src = []
            for i, word_list in enumerate(token_src):
                sub2word_map_src += [i for x in word_list]
            sub2word_map_tgt = []
            for i, word_list in enumerate(token_tgt):
                sub2word_map_tgt += [i for x in word_list]

            sub2word_map_srcs.append(sub2word_map_src)
            sub2word_map_tgts.append(sub2word_map_tgt)

        input_ids_src, lens_src = padding_sequence(input_ids_src, return_len=True)
        attention_mask_src = padding_sequence(attention_mask_src)

        input_ids_tgt, lens_tgt = padding_sequence(input_ids_tgt, return_len=True)
        attention_mask_tgt = padding_sequence(attention_mask_tgt)

        out_src = self._model(np.array(input_ids_src), attention_mask=np.array(
            attention_mask_src), output_hidden_states=True).hidden_states
        out_tgt = self._model(np.array(input_ids_tgt), attention_mask=np.array(
            attention_mask_tgt), output_hidden_states=True).hidden_states
        out_src = out_src[align_layer]
        out_tgt = out_tgt[align_layer]

        aligns = []
        for i in range(len(out_src)):
            dot_product = tf.matmul(out_src[i, :lens_src[i]][1:-1], tf.transpose(out_tgt[i, :lens_tgt[i]][1:-1]))
            softmax_srctgt = tf.nn.softmax(dot_product, axis=-1)
            softmax_tgtsrc = tf.nn.softmax(dot_product, axis=-2)
            softmax_inter = tf.cast(softmax_srctgt > threshold, tf.float32) * \
                tf.cast(softmax_tgtsrc > threshold, tf.float32)
            align_words = set()
            for k, j in np.array(np.nonzero(softmax_inter)).T:
                align_words.add((sub2word_map_srcs[i][k], sub2word_map_tgts[i][j]))

            aligns.append([(i, j) for i, j in sorted(align_words)])
        return aligns
