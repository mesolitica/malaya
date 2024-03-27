import re
import logging
from unidecode import unidecode
from collections import defaultdict

logger = logging.getLogger(__name__)


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


def parse(string):
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
            logger.warning(e)

    return results, ' '.join(last_object)


def rebel_format(article):
    prev_len = 0
    count = 0
    for text_paragraph in article['text'].split('\n'):
        if len(text_paragraph) == 0:
            continue
        sentences = re.split(r'(?<=[.])\s', text_paragraph)
        text = ''
        for sentence in sentences:
            text += sentence + ' '
            if any([entity['boundaries'][0] < len(text) + prev_len < entity['boundaries'][1]
                   for entity in article['entities']]):
                continue
            entities = sorted([entity for entity in article['entities'] if prev_len <
                              entity['boundaries'][1] <= len(text)+prev_len], key=lambda tup: tup['boundaries'][0])
            decoder_output = '<triplet> '
            for int_ent, entity in enumerate(entities):
                triplets = sorted([triplet for triplet in article['triples'] if triplet['subject'] == entity and prev_len < triplet['subject']['boundaries'][1] <= len(
                    text) + prev_len and prev_len < triplet['object']['boundaries'][1] <= len(text) + prev_len], key=lambda tup: tup['object']['boundaries'][0])
                if len(triplets) == 0:
                    continue
                decoder_output += entity['surfaceform'] + ' <subj> '
                for triplet in triplets:
                    decoder_output += triplet['object']['surfaceform'] + \
                        ' <obj> ' + triplet['predicate']['surfaceform'] + ' <subj> '
                decoder_output = decoder_output[:-len(' <subj> ')]
                decoder_output += ' <triplet> '
            decoder_output = decoder_output[:-len(' <triplet> ')]
            count += 1
            prev_len += len(text)

            if len(decoder_output) == 0:
                text = ''
                continue

            text = re.sub('([\\[\\].,!?()])', r' \1 ', text.replace('()', ''))
            text = re.sub('\\s{2,}', ' ', text)

            a = {
                'title': article['title'],
                'context': text,
                'id': article['uri'] + '-' + str(count),
                'triplets': decoder_output,
            }
            return a


def parse_rebel(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace('<s>', '').replace("<pad>", '').replace('</s>', '').split():
        if token == '<triplet>':
            current = 't'
            if relation != '':
                triplets.append(
                    {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == '<subj>':
            current = 's'
            if relation != '':
                triplets.append(
                    {'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
            object_ = ''
        elif token == '<obj>':
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(),
                         'type': relation.strip(),
                         'tail': object_.strip()})
    return triplets


def dict_to_list(triplets):
    q = []
    for no, triple in enumerate(triplets):
        q.append([triple['head'], triple['type'], triple['tail']])
    return q


def rebel_format(triplets):
    """
    Convert
    [['Bruno Santana', 'participant of', '2004 Summer Olympics'],
    ['Bruno Santana', 'participant of', '2008 Summer Olympics'],
    ['Bruno Santana', 'country of citizenship', 'Brazil']]
    to rebel format,
    <triplet> Bruno Santana <subj> 2004 Summer Olympics <obj> participant of <subj> 2008 Summer Olympics <obj> participant of <subj> Brazil <obj> country of citizenship
    """
    q = []
    for no, triple in enumerate(triplets):
        obj = ['<obj>'] + triple[1].split()
        subj = ['<subj>'] + triple[2].split()
        if no > 0 and triple[0] == triplets[no - 1][0]:
            q.extend(subj + obj)
        else:
            triplet = ['<triplet>'] + triple[0].split()
            q.extend(triplet + subj + obj)
    return ' '.join(q)
