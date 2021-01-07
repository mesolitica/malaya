import collections
import re

MaskedLmInstance = collections.namedtuple(
    'MaskedLmInstance', ['index', 'label']
)


def is_number_regex(s):
    if re.match('^\d+?\.\d+?$', s) is None:
        return s.isdigit()
    return True


def reject(token):
    t = token.replace('##', '')
    if is_number_regex(t):
        return True
    if t.startswith('RM'):
        return True
    if token in '!{<>}:;.,"\'':
        return True
    return False


def create_masked_lm_predictions(
    tokens,
    vocab_words,
    rng,
    max_predictions_per_seq = 20,
    masked_lm_prob = 0.1,
    do_whole_word_mask = True,
):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]' or token == '[MASK2]':
            continue
        if reject(token):
            continue
        if (
            do_whole_word_mask
            and len(cand_indexes) >= 1
            and token.startswith('##')
        ):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(tokens) * masked_lm_prob))),
    )

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = '[MASK]'
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[
                        np.random.randint(0, len(vocab_words) - 1)
                    ]

            output_tokens[index] = masked_token

            masked_lms.append(
                MaskedLmInstance(index = index, label = tokens[index])
            )
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key = lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)
