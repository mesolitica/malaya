import tensorflow as tf
import sentencepiece as spm
import model as modeling
import optimization
from prepro_utils import preprocess_text, encode_ids, encode_pieces
from tensorflow.python.estimator.run_config import RunConfig

learning_rate = 5e-5
num_train_steps = 500000
num_warmup_steps = 10000
batch_size = 256

sp_model = spm.SentencePieceProcessor()
sp_model.Load('sp10m.cased.bert.model')

with open('sp10m.cased.bert.vocab') as fopen:
    v = fopen.read().split('\n')[:-1]
v = [i.split('\t') for i in v]
v = {i[0]: i[1] for i in v}


def get_masked_lm_output(
    bert_config,
    input_tensor,
    output_weights,
    positions,
    label_ids,
    label_weights,
):
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope('cls/predictions'):
        with tf.variable_scope('transform'):
            input_tensor = tf.layers.dense(
                input_tensor,
                units = bert_config.hidden_size,
                activation = modeling.gelu,
                kernel_initializer = modeling.create_initializer(0.02),
            )
            input_tensor = modeling.layer_norm(input_tensor)
        output_bias = tf.get_variable(
            'output_bias',
            shape = [bert_config.vocab_size],
            initializer = tf.zeros_initializer(),
        )
        logits = tf.matmul(input_tensor, output_weights, transpose_b = True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis = -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth = bert_config.vocab_size, dtype = tf.float32
        )
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis = [-1]
        )
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    with tf.variable_scope('cls/seq_relationship'):
        output_weights = tf.get_variable(
            'output_weights',
            shape = [2, bert_config.hidden_size],
            initializer = modeling.create_initializer(0.02),
        )
        output_bias = tf.get_variable(
            'output_bias', shape = [2], initializer = tf.zeros_initializer()
        )

        logits = tf.matmul(input_tensor, output_weights, transpose_b = True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis = -1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth = 2, dtype = tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis = -1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank = 3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype = tf.int32) * seq_length, [-1, 1]
    )
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(
        sequence_tensor, [batch_size * seq_length, width]
    )
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def model_fn(features, labels, mode, params):

    input_ids = features['input_ids']
    segment_ids = features['segment_ids']
    masked_lm_positions = features['masked_lm_positions']
    masked_lm_ids = features['masked_lm_ids']
    masked_lm_weights = features['masked_lm_weights']
    next_sentence_labels = features['next_sentence_labels']

    model = modeling.Model(
        dim = 768, vocab_size = 32000, depth = 12, mlp_dim = 768
    )
    sequence_output = model(
        x = input_ids, token_type_ids = segment_ids, training = True
    )

    (
        masked_lm_loss,
        masked_lm_example_loss,
        masked_lm_log_probs,
    ) = get_masked_lm_output(
        model,
        sequence_output,
        model.embedding_table,
        masked_lm_positions,
        masked_lm_ids,
        masked_lm_weights,
    )

    (
        next_sentence_loss,
        next_sentence_example_loss,
        next_sentence_log_probs,
    ) = get_next_sentence_output(
        model, model.pooled_output, next_sentence_labels
    )

    total_loss = masked_lm_loss + next_sentence_loss

    tf.identity(total_loss, 'total_loss')
    tf.identity(masked_lm_loss, 'masked_lm_loss')
    tf.identity(next_sentence_loss, 'next_sentence_loss')

    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('masked_lm_loss', masked_lm_loss)
    tf.summary.scalar('next_sentence_loss', next_sentence_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            total_loss, learning_rate, num_train_steps, num_warmup_steps, False
        )
        estimator_spec = tf.estimator.EstimatorSpec(
            mode = mode, loss = loss, train_op = train_op
        )

    elif mode == tf.estimator.ModeKeys.EVAL:

        def metric_fn(
            masked_lm_example_loss,
            masked_lm_log_probs,
            masked_lm_ids,
            masked_lm_weights,
            next_sentence_example_loss,
            next_sentence_log_probs,
            next_sentence_labels,
        ):
            masked_lm_log_probs = tf.reshape(
                masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]]
            )
            masked_lm_predictions = tf.argmax(
                masked_lm_log_probs, axis = -1, output_type = tf.int32
            )
            masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
            masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
            masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
            masked_lm_accuracy = tf.metrics.accuracy(
                labels = masked_lm_ids,
                predictions = masked_lm_predictions,
                weights = masked_lm_weights,
            )
            masked_lm_mean_loss = tf.metrics.mean(
                values = masked_lm_example_loss, weights = masked_lm_weights
            )

            next_sentence_log_probs = tf.reshape(
                next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]]
            )
            next_sentence_predictions = tf.argmax(
                next_sentence_log_probs, axis = -1, output_type = tf.int32
            )
            next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
            next_sentence_accuracy = tf.metrics.accuracy(
                labels = next_sentence_labels,
                predictions = next_sentence_predictions,
            )
            next_sentence_mean_loss = tf.metrics.mean(
                values = next_sentence_example_loss
            )

            return {
                'masked_lm_accuracy': masked_lm_accuracy,
                'masked_lm_loss': masked_lm_mean_loss,
                'next_sentence_accuracy': next_sentence_accuracy,
                'next_sentence_loss': next_sentence_mean_loss,
            }

        eval_metrics = (
            metric_fn,
            [
                masked_lm_example_loss,
                masked_lm_log_probs,
                masked_lm_ids,
                masked_lm_weights,
                next_sentence_example_loss,
                next_sentence_log_probs,
                next_sentence_labels,
            ],
        )

        estimator_spec = tf.estimator.EstimatorSpec(
            mode = tf.estimator.ModeKeys.EVAL,
            loss = total_loss,
            eval_metrics = eval_metrics,
        )

    return estimator_spec


def run_training(
    train_fn,
    model_fn,
    model_dir: str,
    gpu_mem_fraction: float = 0.96,
    log_step: int = 100,
    summary_step: int = 100,
    save_checkpoint_step: int = 1000,
    max_steps: int = 10000,
    eval_step: int = 10,
    eval_throttle: int = 120,
    train_batch_size: int = 128,
    train_hooks = None,
    eval_fn = None,
):
    tf.logging.set_verbosity(tf.logging.INFO)
    dist_strategy = None

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction = gpu_mem_fraction
    )
    config = tf.ConfigProto(
        allow_soft_placement = True, gpu_options = gpu_options
    )
    run_config = RunConfig(
        train_distribute = dist_strategy,
        eval_distribute = dist_strategy,
        log_step_count_steps = log_step,
        model_dir = model_dir,
        save_checkpoints_steps = save_checkpoint_step,
        save_summary_steps = summary_step,
        session_config = config,
    )

    estimator = tf.estimator.Estimator(
        model_fn = model_fn, params = {}, config = run_config
    )

    if eval_fn:
        train_spec = tf.estimator.TrainSpec(
            input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn = eval_fn, steps = eval_step, throttle_secs = eval_throttle
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    else:
        estimator.train(
            input_fn = train_fn, max_steps = max_steps, hooks = train_hooks
        )


train_hooks = [
    tf.train.LoggingTensorHook(
        ['total_loss', 'masked_lm_loss', 'next_sentence_loss'], every_n_iter = 1
    )
]

train_dataset = get_dataset()

save_directory = 'fnet-base'

train.run_training(
    train_fn = train_dataset,
    model_fn = model_fn,
    model_dir = save_directory,
    log_step = 1,
    save_checkpoint_step = 20000,
    max_steps = total_steps,
    train_hooks = train_hooks,
    eval_step = 0,
    eval_fn = train_dataset,
)
