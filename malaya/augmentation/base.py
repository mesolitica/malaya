def _make_upper(p, o):
    p_split = p.split()
    o_split = o.split()
    return ' '.join(
        [
            s.title() if o_split[no][0].isupper() else s
            for no, s in enumerate(p_split)
        ]
    )


def _replace(word, replace_dict, threshold=0.5):
    word = list(word[:])
    for i in range(len(word)):
        if word[i] in replace_dict and random.random() >= threshold:
            word[i] = replace_dict[word[i]]
    return ''.join(word)
