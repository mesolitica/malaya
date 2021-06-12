import re
import logging
from unidecode import unidecode
from collections import defaultdict


def _is_number_regex(s):
    if re.match('^\\d+?\\.\\d+?$', s) is None:
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
ordinals = ['st', 'nd', 'rd', 'th']

selected_words = ['of', '-', 'for', 'and', "'s", "s'", '+', ':', 'bin', 'binti', 'the']


def check_en_ordinal(string):
    return _is_number_regex(string[0]) and any([string.endswith(o) for o in ordinals])


def parse_triples(string):
    splitted = unidecode(string).split(',')
    splitted = [re.sub(r'[ ]+', ' ', s).strip() for s in splitted]
    splitted = [s[:-1] if s[-1] == '.' else s for s in splitted]
    last_object = None
    results = []
    for no, string in enumerate(splitted):
        try:
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
                    if s[0] == '(':
                        while no_ < len(string) and string[no_][-1] != ')':
                            tokens[index].append(string[no_])
                            no_ += 1
                        tokens[index].append(string[no_])
                    elif tokens[index][0][0].isupper() and s[0].isupper():
                        tokens[index].append(s)
                    elif tokens[index][0][0].islower() and s[0].islower():
                        tokens[index].append(s)
                    elif _is_number_regex(
                        tokens[index][0]
                    ) and _is_number_regex(s):
                        tokens[index].append(s)
                    elif _is_number_regex(tokens[index][0]) and s[0].isupper():
                        tokens[index].append(s)
                    elif tokens[index][0][0].isupper() and _is_number_regex(s):
                        tokens[index].append(s)
                    elif check_en_ordinal(tokens[index][0]) and s[0].isupper():
                        tokens[index].append(s)
                    elif s in selected_words:
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

            tokens['subject'] = ' '.join(tokens.pop(0))
            tokens['relation'] = ' '.join(tokens.pop(1))
            tokens['object'] = ' '.join(tokens.pop(2))
            results.append(dict(tokens))
        except Exception as e:
            logging.warning(e)
            pass

    return results, ' '.join(last_object)
