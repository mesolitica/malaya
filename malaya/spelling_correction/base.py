from malaya.text.tatabahasa import (
    consonants,
    vowels,
    group_compound,
    quad_vowels,
)
from malaya.text.tatabahasa import (
    permulaan,
    hujung,
)
from itertools import product


def _get_indices(string, c='a'):
    return [i for i in range(len(string)) if string[i] == c]


def _permutate(string, indices):
    p = [''.join(_set) for _set in product(list(vowels), repeat=len(indices))]
    p = [p_ for p_ in p if not all([a in p_ for a in quad_vowels])]
    mutate = []
    for p_ in p:
        s = list(string[:])
        for i in range(len(indices)):
            s[indices[i]] = p_[i]
        mutate.append(''.join(s))
    return mutate


def _permutate_sp(string, indices, sp_tokenizer):
    p = [''.join(_set) for _set in product(list(vowels), repeat=len(indices))]
    p = [p_ for p_ in p if not all([a in p_ for a in quad_vowels])]
    mutate = []
    for p_ in p:
        s = list(string[:])
        for i in range(len(indices)):
            s[indices[i]] = p_[i]
        s = ''.join(s)
        if sp_tokenizer.tokenize(s)[0] == '▁':
            continue
        mutate.append(s)
    return mutate


def _augment_vowel_alternate(string):
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
            if c + string[i] in group_compound and not last_time:
                r.append(c + string[i])
                last_time = True
            elif string[last] + c in group_compound and not last_time:
                r.append(string[last] + c)
                last_time = True
            else:
                last_time = False
                if len(r):
                    # ['ng'] gg
                    if (
                        r[-1] in group_compound
                        and c + string[i] == r[-1][-1] * 2
                    ):
                        r.append('^')
                        continue
                    elif r[-1] in group_compound and c == r[-1][-1]:
                        if c + string[i] in group_compound:
                            continue
                        else:
                            r.append('a')
                            continue
                r.append(c + 'a')

        else:
            if len(r):
                if r[-1] in group_compound and c == r[-1][-1]:
                    continue
            r.append(c)

    if len(r):

        if r[-1][-1] in vowels and string[-1] in consonants:
            r.append(string[-1])

        elif (
            r[-1] in group_compound
            and string[-2] in vowels
            and string[-1] in consonants
        ):
            r.append(string[-2:])

    left = ''.join(r).replace('^', '')
    right = left + 'a'
    if string[-1] in vowels:
        left = left + string[-1]
    return left, right


def _augment_vowel_prob(word, **kwargs):
    l, r = _augment_vowel_alternate(word)
    return list(
        set(_permutate(l, _get_indices(l)) + _permutate(r, _get_indices(r)))
    )


def _augment_vowel_prob_sp(word, sp_tokenizer):
    l, r = _augment_vowel_alternate(word)
    return list(
        set(
            _permutate_sp(l, _get_indices(l), sp_tokenizer)
            + _permutate_sp(r, _get_indices(r), sp_tokenizer)
        )
    )


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


def get_permulaan_hujung(word):
    cp_word = word[:]
    hujung_result = [(k, v) for k, v in hujung.items() if word.endswith(k)]
    if len(hujung_result):
        hujung_result = max(hujung_result, key=lambda x: len(x[1]))
        word = word[: -len(hujung_result[0])]
        hujung_result = hujung_result[1]

    permulaan_result = [
        (k, v) for k, v in permulaan.items() if word.startswith(k)
    ]
    if len(permulaan_result):
        permulaan_result = max(permulaan_result, key=lambda x: len(x[1]))
        word = word[len(permulaan_result[0]):]
        permulaan_result = permulaan_result[1]

    return word, hujung_result, permulaan_result
