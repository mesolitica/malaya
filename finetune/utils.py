import tensorflow as tf
from tensorflow.python.distribute.cross_device_ops import (
    AllReduceCrossDeviceOps,
)
from tensorflow.python.estimator.run_config import RunConfig
import collections
import re


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match('^(.*):\\d+$', name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue

        assignment_map[name] = name
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ':0'] = 1

    tf.logging.info('**** Trainable Variables ****')
    for var in tvars:
        init_string = ''
        if var.name in initialized_variable_names:
            init_string = ', *INIT_FROM_CKPT*'
        tf.logging.info(
            '  name = %s, shape = %s%s', var.name, var.shape, init_string
        )

    return (assignment_map, initialized_variable_names)


def run_training(
    train_fn,
    model_fn,
    model_dir: str,
    num_gpus: int = 1,
    log_step: int = 100,
    summary_step: int = 100,
    save_checkpoint_step: int = 1000,
    max_steps: int = 10000,
    eval_step: int = 10,
    eval_throttle: int = 120,
    train_hooks = None,
    eval_fn = None,
):
    tf.logging.set_verbosity(tf.logging.INFO)

    if num_gpus > 1 and not use_tpu:
        dist_strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus = num_gpus,
            auto_shard_dataset = True,
            cross_device_ops = AllReduceCrossDeviceOps(
                'nccl', num_packs = num_gpus
            ),
        )
    else:
        dist_strategy = None

    run_config = RunConfig(
        train_distribute = dist_strategy,
        eval_distribute = dist_strategy,
        log_step_count_steps = log_step,
        model_dir = model_dir,
        save_checkpoints_steps = save_checkpoint_step,
        save_summary_steps = summary_step,
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
