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
        string = string.split()
        no_ = 0
        while no_ < len(string):
            s = string[no_]
            s = re.sub(r'-+', '-', s)
            if len(tokens[index]) == 0:
                tokens[index].append(s)
            else:
                if s == '(':
                    tokens[index].append(s)
                    no_ += 1
                    while no_ < len(string):
                        tokens[index].append(string[no_])
                        if string[no_] == ')':
                            break
                        no_ += 1
                elif tokens[index][0][0].isupper() and s[0].isupper():
                    tokens[index].append(s)
                elif tokens[index][0][0].islower() and s[0].islower():
                    tokens[index].append(s)
                elif _is_number_regex(tokens[index][0]) and _is_number_regex(s):
                    tokens[index].append(s)
                elif _is_number_regex(tokens[index][0]) and s[0].isupper():
                    tokens[index].append(s)
                elif s in ['of', '-', 'for', 'and', "'s", "s'"]:
                    tokens[index].append(s)
                elif s.lower() in months:
                    tokens[index].append(s)
                else:
                    index += 1
                    tokens[index].append(s)
            no_ += 1

        if no == 0:
            last_object = tokens[0]

        if len(tokens) > 3:
            for i in range(3, len(tokens), 1):
                tokens[2].extend(tokens[i])
                tokens.pop(i, None)

        if len(tokens) == 2:
            tokens[2] = [tokens[1][-1]]
            tokens[1] = tokens[1][:-1]

        results.append(tokens)

    return results, last_object
