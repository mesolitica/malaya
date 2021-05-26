import re
from collections import defaultdict


def _is_number_regex(s):
    if re.match('^\d+?\.\d+?$', s) is None:
        return s.isdigit()
    return True


months = [
    'january',
    'jan',
    'february',
    'feb',
    'march',
    'april',
    'apr',
    'may',
    'june',
    'july',
    'august',
    'aug',
    'september',
    'sep',
    'october',
    'oct',
    'november',
    'nov',
    'december',
    'dec',
]


def parse_triples(string):
    splitted = string.split(',')
    splitted = [re.sub(r'[ ]+', ' ', s).strip() for s in splitted]
    last_object = None
    results = []
    for no, string in enumerate(splitted):
        index = 0
        tokens = defaultdict(list)
        if string[0].islower():
            tokens[0].extend(last_object)
            index += 1
        for s in string.split():
            s = re.sub(r'-+', '-', s)
            if len(tokens[index]) == 0:
                tokens[index].append(s)
            else:
                if tokens[index][0][0].isupper() and s[0].isupper():
                    tokens[index].append(s)
                elif tokens[index][0][0].islower() and s[0].islower():
                    tokens[index].append(s)
                elif _is_number_regex(tokens[index][0]) and _is_number_regex(s):
                    tokens[index].append(s)
                elif _is_number_regex(tokens[index][0]) and s[0].isupper():
                    tokens[index].append(s)
                elif tokens[index][-1] in ['('] and _is_number_regex(s):
                    tokens[index].append(s)
                elif s in ['of', '-', '(', ')']:
                    tokens[index].append(s)
                elif s.lower() in months:
                    tokens[index].append(s)
                else:
                    index += 1
                    tokens[index].append(s)
        if no == 0:
            last_object = tokens[0]

        if len(tokens) > 3:
            for i in range(3, len(tokens), 1):
                tokens[2].extend(tokens[i])
                tokens.pop(i, None)

        results.append(tokens)
    return results
