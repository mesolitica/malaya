import collections
import numpy as np
import math
import re
import six
import gc
from malaya.text.bpe import (
    encode_sentencepiece_ids,
    preprocess_text,
    encode_sentencepiece,
    SPIECE_UNDERLINE,
    SEG_ID_P,
    SEG_ID_Q,
    SEG_ID_CLS,
    SEG_ID_SEP,
    SEG_ID_PAD,
    SEP_ID,
    CLS_ID,
)

RawResultV2 = collections.namedtuple(
    'RawResultV2',
    [
        'unique_id',
        'start_top_log_probs',
        'start_top_index',
        'end_top_log_probs',
        'end_top_index',
        'cls_logits',
    ],
)

_PrelimPrediction = collections.namedtuple(
    'PrelimPrediction',
    [
        'feature_index',
        'start_index',
        'end_index',
        'start_log_prob',
        'end_log_prob',
    ],
)

_NbestPrediction = collections.namedtuple(
    'NbestPrediction',
    [
        'text',
        'start_log_prob',
        'end_log_prob',
        'start_orig_pos',
        'end_orig_pos',
    ],
)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id,
        example_index,
        doc_span_index,
        tok_start_to_orig_index,
        tok_end_to_orig_index,
        token_is_max_context,
        tokens,
        input_ids,
        input_mask,
        p_mask,
        segment_ids,
        paragraph_len,
        cls_index=None,
        start_position=None,
        end_position=None,
        is_impossible=None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tok_start_to_orig_index = tok_start_to_orig_index
        self.tok_end_to_orig_index = tok_end_to_orig_index
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.p_mask = p_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.cls_index = cls_index
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class SquadExample(object):
    def __init__(
        self,
        question_text,
        paragraph_text,
        orig_answer_text=None,
        start_position=None,
        end_position=None,
        is_impossible=False,
    ):
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__dict__


def _convert_index(index, pos, m=None, is_start=True):
    """Converts index."""
    if index[pos] is not None:
        return index[pos]
    n = len(index)
    rear = pos
    while rear < n - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if m is not None and index[front] < m - 1:
            if is_start:
                return index[front] + 1
            else:
                return m - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = (
            min(num_left_context, num_right_context) + 0.01 * doc_span.length
        )
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def convert_examples_to_features_xlnet(
    examples,
    tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
):

    cnt_pos, cnt_neg = 0, 0
    unique_id = 1000000000
    max_N, max_M = 1024, 1024
    f = np.zeros((max_N, max_M), dtype=np.float32)
    features = []

    for n in range(len(examples)):
        example_index = n
        example = examples[n]

        query_tokens = encode_sentencepiece_ids(
            tokenizer, preprocess_text(example.question_text, lower=False)
        )

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        paragraph_text = example.paragraph_text
        para_tokens = encode_sentencepiece(
            tokenizer.sp_model, preprocess_text(example.paragraph_text, lower=False)
        )

        chartok_to_tok_index = []
        tok_start_to_chartok_index = []
        tok_end_to_chartok_index = []
        char_cnt = 0
        for i, token in enumerate(para_tokens):
            chartok_to_tok_index.extend([i] * len(token))
            tok_start_to_chartok_index.append(char_cnt)
            char_cnt += len(token)
            tok_end_to_chartok_index.append(char_cnt - 1)

        tok_cat_text = ''.join(para_tokens).replace(SPIECE_UNDERLINE, ' ')
        N, M = len(paragraph_text), len(tok_cat_text)

        if N > max_N or M > max_M:
            max_N = max(N, max_N)
            max_M = max(M, max_M)
            f = np.zeros((max_N, max_M), dtype=np.float32)
            gc.collect()

        g = {}

        def _lcs_match(max_dist):
            f.fill(0)
            g.clear()
            for i in range(N):
                for j in range(i - max_dist, i + max_dist):
                    if j >= M or j < 0:
                        continue

                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]

                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]

                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    if (
                        preprocess_text(
                            paragraph_text[i],
                            lower=False,
                            remove_space=False,
                        )
                        == tok_cat_text[j]
                        and f_prev + 1 > f[i, j]
                    ):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1

        max_dist = abs(N - M) + 5
        for _ in range(2):
            _lcs_match(max_dist)
            if f[N - 1, M - 1] > 0.8 * N:
                break
            max_dist *= 2

        orig_to_chartok_index = [None] * N
        chartok_to_orig_index = [None] * M
        i, j = N - 1, M - 1
        while i >= 0 and j >= 0:
            if (i, j) not in g:
                break
            if g[(i, j)] == 2:
                orig_to_chartok_index[i] = j
                chartok_to_orig_index[j] = i
                i, j = i - 1, j - 1
            elif g[(i, j)] == 1:
                j = j - 1
            else:
                i = i - 1

        if (
            all(v is None for v in orig_to_chartok_index)
            or f[N - 1, M - 1] < 0.8 * N
        ):
            continue

        tok_start_to_orig_index = []
        tok_end_to_orig_index = []
        for i in range(len(para_tokens)):
            start_chartok_pos = tok_start_to_chartok_index[i]
            end_chartok_pos = tok_end_to_chartok_index[i]
            start_orig_pos = _convert_index(
                chartok_to_orig_index, start_chartok_pos, N, is_start=True
            )
            end_orig_pos = _convert_index(
                chartok_to_orig_index, end_chartok_pos, N, is_start=False
            )

            tok_start_to_orig_index.append(start_orig_pos)
            tok_end_to_orig_index.append(end_orig_pos)

        if not is_training:
            tok_start_position = tok_end_position = None

        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1

        if is_training and not example.is_impossible:
            start_position = example.start_position
            end_position = start_position + len(example.orig_answer_text) - 1

            start_chartok_pos = _convert_index(
                orig_to_chartok_index, start_position, is_start=True
            )
            tok_start_position = chartok_to_tok_index[start_chartok_pos]

            end_chartok_pos = _convert_index(
                orig_to_chartok_index, end_position, is_start=False
            )
            tok_end_position = chartok_to_tok_index[end_chartok_pos]
            assert tok_start_position <= tok_end_position

        def _piece_to_id(x):
            if six.PY2 and isinstance(x, unicode):
                x = x.encode('utf-8')
            return tokenizer.sp_model.PieceToId(x)

        all_doc_tokens = list(map(_piece_to_id, para_tokens))

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = collections.namedtuple('DocSpan', ['start', 'length'])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_is_max_context = {}
            segment_ids = []
            p_mask = []

            cur_tok_start_to_orig_index = []
            cur_tok_end_to_orig_index = []

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i

                cur_tok_start_to_orig_index.append(
                    tok_start_to_orig_index[split_token_index]
                )
                cur_tok_end_to_orig_index.append(
                    tok_end_to_orig_index[split_token_index]
                )

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index
                )
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(SEG_ID_P)
                p_mask.append(0)

            paragraph_len = len(tokens)

            tokens.append(SEP_ID)
            segment_ids.append(SEG_ID_P)
            p_mask.append(1)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(SEG_ID_Q)
                p_mask.append(1)
            tokens.append(SEP_ID)
            segment_ids.append(SEG_ID_Q)
            p_mask.append(1)

            cls_index = len(segment_ids)
            tokens.append(CLS_ID)
            segment_ids.append(SEG_ID_CLS)
            p_mask.append(0)

            input_ids = tokens
            input_mask = [0] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(1)
                segment_ids.append(SEG_ID_PAD)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(p_mask) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (
                    tok_start_position >= doc_start
                    and tok_end_position <= doc_end
                ):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = 0
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 20:

                if is_training and span_is_impossible:
                    pass

                if is_training and not span_is_impossible:
                    pieces = [
                        tokenizer.sp_model.IdToPiece(token)
                        for token in tokens[start_position: (end_position + 1)]
                    ]
                    answer_text = tokenizer.sp_model.DecodePieces(pieces)
            if is_training:
                feat_example_index = None
            else:
                feat_example_index = example_index

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=feat_example_index,
                doc_span_index=doc_span_index,
                tok_start_to_orig_index=cur_tok_start_to_orig_index,
                tok_end_to_orig_index=cur_tok_end_to_orig_index,
                token_is_max_context=token_is_max_context,
                tokens=[tokenizer.sp_model.IdToPiece(x) for x in tokens],
                input_ids=input_ids,
                input_mask=input_mask,
                p_mask=p_mask,
                segment_ids=segment_ids,
                paragraph_len=paragraph_len,
                cls_index=cls_index,
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
            )

            features.append(feature)

            unique_id += 1
            if span_is_impossible:
                cnt_neg += 1
            else:
                cnt_pos += 1
    return features


def convert_examples_to_features_bert(
    examples,
    tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    is_training=False,
    do_lower_case=False,
):
    """Loads a data file into a list of `InputBatch`s."""

    cnt_pos, cnt_neg = 0, 0
    unique_id = 1000000000
    max_n, max_m = 1024, 1024
    f = np.zeros((max_n, max_m), dtype=np.float32)
    features = []

    for n in range(len(examples)):
        example_index = n
        example = examples[n]

        query_tokens = encode_sentencepiece_ids(
            tokenizer,
            preprocess_text(example.question_text, lower=do_lower_case),
        )

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        paragraph_text = example.paragraph_text
        para_tokens = encode_sentencepiece(
            tokenizer.sp_model,
            preprocess_text(example.paragraph_text, lower=do_lower_case),
            return_unicode=False,
        )

        chartok_to_tok_index = []
        tok_start_to_chartok_index = []
        tok_end_to_chartok_index = []
        char_cnt = 0
        para_tokens = [six.ensure_text(token, 'utf-8') for token in para_tokens]
        for i, token in enumerate(para_tokens):
            new_token = six.ensure_text(token).replace(SPIECE_UNDERLINE, ' ')
            chartok_to_tok_index.extend([i] * len(new_token))
            tok_start_to_chartok_index.append(char_cnt)
            char_cnt += len(new_token)
            tok_end_to_chartok_index.append(char_cnt - 1)

        tok_cat_text = ''.join(para_tokens).replace(SPIECE_UNDERLINE, ' ')
        n, m = len(paragraph_text), len(tok_cat_text)

        if n > max_n or m > max_m:
            max_n = max(n, max_n)
            max_m = max(m, max_m)
            f = np.zeros((max_n, max_m), dtype=np.float32)

        g = {}

        def _lcs_match(max_dist, n=n, m=m):
            f.fill(0)
            g.clear()
            for i in range(n):
                for j in range(i - max_dist, i + max_dist):
                    if j >= m or j < 0:
                        continue

                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]

                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]

                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    if (
                        preprocess_text(
                            paragraph_text[i],
                            lower=do_lower_case,
                            remove_space=False,
                        )
                        == tok_cat_text[j]
                        and f_prev + 1 > f[i, j]
                    ):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1

        max_dist = abs(n - m) + 5
        for _ in range(2):
            _lcs_match(max_dist)
            if f[n - 1, m - 1] > 0.8 * n:
                break
            max_dist *= 2

        orig_to_chartok_index = [None] * n
        chartok_to_orig_index = [None] * m
        i, j = n - 1, m - 1
        while i >= 0 and j >= 0:
            if (i, j) not in g:
                break
            if g[(i, j)] == 2:
                orig_to_chartok_index[i] = j
                chartok_to_orig_index[j] = i
                i, j = i - 1, j - 1
            elif g[(i, j)] == 1:
                j = j - 1
            else:
                i = i - 1

        if (
            all(v is None for v in orig_to_chartok_index)
            or f[n - 1, m - 1] < 0.8 * n
        ):
            continue

        tok_start_to_orig_index = []
        tok_end_to_orig_index = []
        for i in range(len(para_tokens)):
            start_chartok_pos = tok_start_to_chartok_index[i]
            end_chartok_pos = tok_end_to_chartok_index[i]
            start_orig_pos = _convert_index(
                chartok_to_orig_index, start_chartok_pos, n, is_start=True
            )
            end_orig_pos = _convert_index(
                chartok_to_orig_index, end_chartok_pos, n, is_start=False
            )

            tok_start_to_orig_index.append(start_orig_pos)
            tok_end_to_orig_index.append(end_orig_pos)

        if not is_training:
            tok_start_position = tok_end_position = None

        if is_training and example.is_impossible:
            tok_start_position = 0
            tok_end_position = 0

        if is_training and not example.is_impossible:
            start_position = example.start_position
            end_position = start_position + len(example.orig_answer_text) - 1

            start_chartok_pos = _convert_index(
                orig_to_chartok_index, start_position, is_start=True
            )
            tok_start_position = chartok_to_tok_index[start_chartok_pos]

            end_chartok_pos = _convert_index(
                orig_to_chartok_index, end_position, is_start=False
            )
            tok_end_position = chartok_to_tok_index[end_chartok_pos]
            assert tok_start_position <= tok_end_position

        def _piece_to_id(x):
            if six.PY2 and isinstance(x, six.text_type):
                x = six.ensure_binary(x, 'utf-8')
            return tokenizer.sp_model.PieceToId(x)

        all_doc_tokens = list(map(_piece_to_id, para_tokens))
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = collections.namedtuple('DocSpan', ['start', 'length'])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_is_max_context = {}
            segment_ids = []
            p_mask = []

            cur_tok_start_to_orig_index = []
            cur_tok_end_to_orig_index = []

            tokens.append(tokenizer.sp_model.PieceToId('[CLS]'))
            segment_ids.append(0)
            p_mask.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
                p_mask.append(1)
            tokens.append(tokenizer.sp_model.PieceToId('[SEP]'))
            segment_ids.append(0)
            p_mask.append(1)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i

                cur_tok_start_to_orig_index.append(
                    tok_start_to_orig_index[split_token_index]
                )
                cur_tok_end_to_orig_index.append(
                    tok_end_to_orig_index[split_token_index]
                )

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index
                )
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
                p_mask.append(0)
            tokens.append(tokenizer.sp_model.PieceToId('[SEP]'))
            segment_ids.append(1)
            p_mask.append(1)

            paragraph_len = len(tokens)
            input_ids = tokens
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (
                    tok_start_position >= doc_start
                    and tok_end_position <= doc_end
                ):
                    out_of_span = True
                if out_of_span:
                    # continue
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 20:
                if is_training and not span_is_impossible:
                    pieces = [
                        tokenizer.sp_model.IdToPiece(token)
                        for token in tokens[start_position: (end_position + 1)]
                    ]
                    answer_text = tokenizer.sp_model.DecodePieces(pieces)
            if is_training:
                feat_example_index = None
            else:
                feat_example_index = example_index

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=feat_example_index,
                doc_span_index=doc_span_index,
                tok_start_to_orig_index=cur_tok_start_to_orig_index,
                tok_end_to_orig_index=cur_tok_end_to_orig_index,
                token_is_max_context=token_is_max_context,
                tokens=[tokenizer.sp_model.IdToPiece(x) for x in tokens],
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                paragraph_len=paragraph_len,
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                p_mask=p_mask,
            )

            features.append(feature)

            unique_id += 1
            if span_is_impossible:
                cnt_neg += 1
            else:
                cnt_pos += 1

    return features


def read_squad_examples(
    paragraph_text,
    question_texts,
    tokenizer,
    max_seq_length=384,
    doc_stride=128,
    max_query_length=64,
    mode='bert',
):
    modes = {
        'bert': convert_examples_to_features_bert,
        'xlnet': convert_examples_to_features_xlnet,
    }

    start_position = None
    orig_answer_text = None
    is_impossible = False
    examples = []
    for question_text in question_texts:
        example = SquadExample(
            question_text=question_text,
            paragraph_text=paragraph_text,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            is_impossible=is_impossible,
        )
        examples.append(example)

    return (
        examples,
        modes.get(mode, modes['bert'])(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
        ),
    )


def accumulate_predictions_bert(
    all_examples,
    all_features,
    all_results,
    n_best_size=20,
    max_answer_length=30,
    start_n_top=5,
    end_n_top=5,
):
    result_dict, cls_dict = {}, {}

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        if example_index not in result_dict:
            result_dict[example_index] = {}
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            if feature.unique_id not in result_dict[example_index]:
                result_dict[example_index][feature.unique_id] = {}
            result = unique_id_to_result[feature.unique_id]
            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            doc_offset = feature.tokens.index('[SEP]') + 1
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * end_n_top + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]
                    if start_index - doc_offset >= len(
                        feature.tok_start_to_orig_index
                    ):
                        continue
                    if start_index - doc_offset < 0:
                        continue
                    if end_index - doc_offset >= len(
                        feature.tok_end_to_orig_index
                    ):
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_idx = start_index - doc_offset
                    end_idx = end_index - doc_offset
                    if (start_idx, end_idx) not in result_dict[example_index][
                        feature.unique_id
                    ]:
                        result_dict[example_index][feature.unique_id][
                            (start_idx, end_idx)
                        ] = []
                    result_dict[example_index][feature.unique_id][
                        (start_idx, end_idx)
                    ].append((start_log_prob, end_log_prob))
        if example_index not in cls_dict:
            cls_dict[example_index] = []
        cls_dict[example_index].append(score_null)
    return result_dict, cls_dict


def write_predictions_bert(
    result_dict,
    cls_dict,
    all_examples,
    all_features,
    all_results,
    n_best_size=20,
    max_answer_length=30,
    null_score_diff_threshold=None,
):
    outputs = []
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        # score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            for ((start_idx, end_idx), logprobs) in result_dict[example_index][
                feature.unique_id
            ].items():
                start_log_prob = 0
                end_log_prob = 0
                for logprob in logprobs:
                    start_log_prob += logprob[0]
                    end_log_prob += logprob[1]
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=feature_index,
                        start_index=start_idx,
                        end_index=end_idx,
                        start_log_prob=start_log_prob / len(logprobs),
                        end_log_prob=end_log_prob / len(logprobs),
                    )
                )

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True,
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_start_to_orig_index = feature.tok_start_to_orig_index
            tok_end_to_orig_index = feature.tok_end_to_orig_index
            start_orig_pos = tok_start_to_orig_index[pred.start_index]
            end_orig_pos = tok_end_to_orig_index[pred.end_index]

            paragraph_text = example.paragraph_text
            final_text = paragraph_text[
                start_orig_pos: end_orig_pos + 1
            ].strip()

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob,
                    start_orig_pos=start_orig_pos,
                    end_orig_pos=end_orig_pos + 1,
                )
            )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text='',
                    start_log_prob=-1e6,
                    end_log_prob=-1e6,
                    start_orig_pos=None,
                    end_orig_pos=None,
                )
            )

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        outputs.append(best_non_null_entry)
    return outputs


def write_predictions_xlnet(
    all_examples,
    all_features,
    all_results,
    n_best_size=20,
    max_answer_length=64,
):

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()
    outputs = []

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive

        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            cur_null_score = result.cls_logits

            # if we could have irrelevant answers, get the min score of irrelevant
            score_null = min(score_null, cur_null_score)

            for i in range(5):
                for j in range(5):
                    start_log_prob = result.start_top_log_probs[i]
                    start_index = result.start_top_index[i]

                    j_index = i * 5 + j

                    end_log_prob = result.end_top_log_probs[j_index]
                    end_index = result.end_top_index[j_index]

                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= feature.paragraph_len - 1:
                        continue
                    if end_index >= feature.paragraph_len - 1:
                        continue

                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_log_prob=start_log_prob,
                            end_log_prob=end_log_prob,
                        )
                    )

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_log_prob + x.end_log_prob),
            reverse=True,
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_start_to_orig_index = feature.tok_start_to_orig_index
            tok_end_to_orig_index = feature.tok_end_to_orig_index
            start_orig_pos = tok_start_to_orig_index[pred.start_index]
            end_orig_pos = tok_end_to_orig_index[pred.end_index]

            paragraph_text = example.paragraph_text
            final_text = paragraph_text[
                start_orig_pos: end_orig_pos + 1
            ].strip()

            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_log_prob=pred.start_log_prob,
                    end_log_prob=pred.end_log_prob,
                    start_orig_pos=start_orig_pos,
                    end_orig_pos=end_orig_pos + 1,
                )
            )

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text='',
                    start_log_prob=-1e6,
                    end_log_prob=-1e6,
                    start_orig_pos=None,
                    end_orig_pos=None,
                )
            )

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_log_prob + entry.end_log_prob)
            if not best_non_null_entry:
                best_non_null_entry = entry

        outputs.append(best_non_null_entry)
    return outputs
