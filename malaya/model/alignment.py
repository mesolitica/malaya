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
eflomal_model.align(left, right) originally ~4 seconds, now ~200 ms.
"""

import numpy as np
from collections import defaultdict
from tempfile import NamedTemporaryFile
from typing import List


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
    def __init__(self, priors_filename):

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
        self._process_priors()

    def _process_priors(self):
        priors_list_dict = defaultdict(list)
        ferf_priors_dict = defaultdict(list)
        ferr_priors_dict = defaultdict(list)
        hmmf_priors = {}
        hmmr_priors = {}
        with open(self._priors_filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                fields = line.rstrip('\n').split('\t')
                try:
                    alpha = float(fields[-1])
                except ValueError:
                    raise ValueError('ERROR: priors file %s line %d contains alpha value of "%s" which is not numeric' % (
                        args.priors_filename, i+1, fields[2]))

                if fields[0] == 'LEX' and len(fields) == 4:
                    priors_list_dict[(fields[1].lower(), fields[2].lower())].append(alpha)
                elif fields[0] == 'HMMF' and len(fields) == 3:
                    hmmf_priors[int(fields[1])] = alpha
                elif fields[0] == 'HMMR' and len(fields) == 3:
                    hmmr_priors[int(fields[1])] = alpha
                elif fields[0] == 'FERF' and len(fields) == 4:
                    ferf_priors_dict[fields[1].lower()].append((int(fields[2]), alpha))
                elif fields[0] == 'FERR' and len(fields) == 4:
                    ferr_priors_dict[fields[1].lower()].append((int(fields[2]), alpha))
                else:
                    raise ValueError('ERROR: priors file %s line %d is invalid ' % (args.priors_filename, i+1))

        self._priors_list_dict = priors_list_dict
        self._ferf_priors_dict = ferf_priors_dict
        self._ferr_priors_dict = ferr_priors_dict
        self._hmmf_priors = hmmf_priors
        self._hmmr_priors = hmmr_priors

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
        verbose: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        """
        align text using eflomal.
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
                alpha = self._priors_list_dict.get((k.lower(), v.lower()), [])
                for a in alpha:
                    priors_indexed[(e, f)] = priors_indexed.get((e, f), 0.0) + a

        ferf_indexed = {}
        for k in src_index:
            e = get_src_index(k)
            tuples = self._ferf_priors_dict.get(k.lower(), [])
            for t in tuples:
                fert = t[0]
                alpha = t[1]
                ferf_indexed[(e, fert)] = \
                    ferf_indexed.get((e, fert), 0.0) + alpha

        ferr_indexed = {}
        for k in trg_index:
            f = get_trg_index(k)
            tuples = self._ferr_priors_dict.get(k.lower(), [])
            for t in tuples:
                fert = t[0]
                alpha = t[1]
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
                    quiet=not verbose,
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

        return {'forward': fwd.split('\n'), 'reverse': rev.split('\n')}
