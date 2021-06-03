import tensorflow as tf
from tensorflow.python.distribute.cross_device_ops import (
    AllReduceCrossDeviceOps,
)
from tensorflow.python.estimator.run_config import RunConfig
from herpetologist import check_type
from typing import List, Dict
import numpy as np
import collections
import re


@check_type
def run_training(
    train_fn,
    model_fn,
    model_dir: str,
    num_gpus: int = 1,
    gpu_mem_fraction: float = 0.95,
    log_step: int = 100,
    summary_step: int = 100,
    save_checkpoint_step: int = 1000,
    max_steps: int = 10000,
    eval_step: int = 10,
    eval_throttle: int = 120,
    train_hooks=None,
    eval_fn=None,
):
    tf.logging.set_verbosity(tf.logging.INFO)

    if num_gpus > 1 and not use_tpu:
        dist_strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=num_gpus,
            auto_shard_dataset=True,
            cross_device_ops=AllReduceCrossDeviceOps(
                'nccl', num_packs=num_gpus
            ),
        )
    else:
        dist_strategy = None

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_mem_fraction
    )
    config = tf.ConfigProto(
        allow_soft_placement=True, gpu_options=gpu_options
    )
    run_config = RunConfig(
        train_distribute=dist_strategy,
        eval_distribute=dist_strategy,
        log_step_count_steps=log_step,
        model_dir=model_dir,
        save_checkpoints_steps=save_checkpoint_step,
        save_summary_steps=summary_step,
        session_config=config,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, params={}, config=run_config
    )

    if eval_fn:
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_fn, max_steps=max_steps, hooks=train_hooks
        )

        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_fn, steps=eval_step, throttle_secs=eval_throttle
        )
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    else:
        estimator.train(
            input_fn=train_fn, max_steps=max_steps, hooks=train_hooks
        )


@check_type
def prepare_dataset(
    generator,
    data_dir: str,
    shards: List[Dict],
    prefix: str = 'dataset',
    shuffle: bool = True,
    already_shuffled: bool = False,
):
    prepare_data.check_shard(shards)
    filepath_fns = {
        'train': prepare_data.training_filepaths,
        'dev': prepare_data.dev_filepaths,
        'test': prepare_data.test_filepaths,
    }

    split_paths = [
        (
            split['split'],
            filepath_fns[split['split']](
                prefix, data_dir, split['shards'], shuffled=already_shuffled
            ),
        )
        for split in shards
    ]
    all_paths = []
    for _, paths in split_paths:
        all_paths.extend(paths)

    prepare_data.generate_files(generator, all_paths)

    if shuffle:
        prepare_data.shuffle_dataset(all_paths)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint, logging=True):
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

    if logging:
        tf.logging.info('**** Trainable Variables ****')
        for var in tvars:
            init_string = ''
            if var.name in initialized_variable_names:
                init_string = ', *INIT_FROM_CKPT*'
            tf.logging.info(
                '  name = %s, shape = %s%s', var.name, var.shape, init_string
            )

    return (assignment_map, initialized_variable_names)


def calculate_parameters(variables):
    return np.sum(
        [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]
    )
