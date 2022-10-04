from malaya.text.tatabahasa import (
    alphabet,
    consonants,
    vowels,
    group_compound,
    quad_vowels,
)
from malaya.text.normalization import get_hujung, get_permulaan
from itertools import product

replace_consonants = {
    'n': 'm',
    'r': 't',
    'g': 'h',
    'j': 'k',
    'k': 'l',
    'd': 's',
    'b': 'n',
    'f': 'g',
    'm': 'n',
}

letters = 'abcdefghijklmnopqrstuvwxyz'


def _get_indices(string, c='a'):
    return [i for i in range(len(string)) if string[i] == c]


def _permutate(string, indices, sp_tokenizer=None, validate_qual_vowels=False):
    p = [''.join(_set) for _set in product(list(vowels), repeat=len(indices))]
    if validate_qual_vowels:
        p = [p_ for p_ in p if not all([a in p_ for a in quad_vowels])]
    mutate = []
    for p_ in p:
        s = list(string)
        for i in range(len(indices)):
            s[indices[i]] = p_[i]
        if sp_tokenizer is not None:
            s = ''.join(s)
            if sp_tokenizer.tokenize(s)[0] == '▁':
                continue
        mutate.append(''.join(s))
    return mutate


def _augment_vowel_alternate(string, use_group_compound=True):
    """
    malaya.spell._augment_vowel_alternate('sngpore')
    -> ('sangapor', 'sangapora')

    malaya.spell._augment_vowel_alternate('kmpung')
    -> ('kmpung', 'kmpunga')

    malaya.spell._augment_vowel_alternate('aym')
    -> ('ayam', 'ayama')
    """
    r = []
    # a flag to not duplicate
    last_time = False
    for i, c in enumerate(string[:-1], 1):
        last = i - 2
        if last < 0:
            last = 0

        # we only want to put a vowel after consonant if next that consonant if not a wovel
        if c in consonants and string[i] not in vowels:
            if use_group_compound and c + string[i] in group_compound and not last_time:
                r.append(c + string[i])
                last_time = True
            elif use_group_compound and string[last] + c in group_compound and not last_time:
                r.append(string[last] + c)
                last_time = True
            else:
                last_time = False
                if len(r):
                    # ['ng'] gg
                    if (
                        use_group_compound and r[-1] in group_compound
                        and c + string[i] == r[-1][-1] * 2
                    ):
                        r.append('^')
                        continue
                    elif use_group_compound and r[-1] in group_compound and c == r[-1][-1]:
                        if c + string[i] in group_compound:
                            continue
                        else:
                            r.append('a')
                            continue
                r.append(c + 'a')

        else:
            if len(r):
                if use_group_compound and r[-1] in group_compound and c == r[-1][-1]:
                    continue
            r.append(c)

    if len(r):

        if r[-1][-1] in vowels and string[-1] in consonants:
            r.append(string[-1])

        elif (
            use_group_compound and r[-1] in group_compound
            and string[-2] in vowels
            and string[-1] in consonants
        ):
            r.append(string[-2:])

    left = ''.join(r).replace('^', '')
    right = left + 'a'
    if string[-1] in vowels:
        left = left + string[-1]
    return left, right


def norvig_method(word, delete_only=False):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R if R[0] in vowels]

    if delete_only:
        return deletes

    transposes = [
        L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1
    ]
    inserts = [L + c + R for L, R in splits for c in alphabet]

    replaces = [L + replace_consonants.get(R[0], R[0]) + R[1:] for L, R in splits if R]
    replaces.extend([L + c + R[1:] for L, R in splits if R for c in letters])

    return splits, deletes, transposes, inserts, replaces


def _augment_vowel_prob(word, add_norvig_method=False, sp_tokenizer=None, **kwargs):
    l, r = _augment_vowel_alternate(word, use_group_compound=True)
    l_, r_ = _augment_vowel_alternate(word, use_group_compound=False)
    groups = [l, r, l_, r_]
    results = []
    for g in groups:
        results.extend(_permutate(g, _get_indices(g), sp_tokenizer))
        g_norvig = norvig_method(g, delete_only=True)
        for s in g_norvig:
            results.extend(_permutate(s, _get_indices(s), sp_tokenizer))

    return list(set(results))


def _augment_vowel(
    string, selected=['a', 'u', 'i', 'e'], included_end=True
):
    pseudo = []
    if included_end:
        end = string[-1]
    else:
        end = ''
    for c in selected:
        pseudo.append(''.join([w + c for w in string[:-1]] + [end]))
    return pseudo


def _return_possible(word, dicts, edits):
    return set(e2 for e1 in edits(word) for e2 in edits(e1) if e2 in dicts)


def _return_known(word, dicts):
    return set(w for w in word if w in dicts)


def get_permulaan_hujung(word, stemmer=None):
    word, hujung_result = get_hujung(word, stemmer=stemmer)
    word, permulaan_result = get_permulaan(word, stemmer=stemmer)

    return word, hujung_result, permulaan_result
