from tensor2tensor.utils import adafactor
import tensorflow as tf
import re


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    # AdafactorOptimizer.beta1 = 0.0
    # AdafactorOptimizer.clipping_threshold = 1.0
    # AdafactorOptimizer.decay_rate = None
    # AdafactorOptimizer.epsilon1 = 1e-30
    # AdafactorOptimizer.epsilon2 = 0.001
    # AdafactorOptimizer.factored = True
    # AdafactorOptimizer.min_dim_size_to_factor = 128
    # AdafactorOptimizer.multiply_by_parameter_scale = True

    global_step = tf.train.get_or_create_global_step()

    optimizer = adafactor.AdafactorOptimizer(
        multiply_by_parameter_scale = True,
        learning_rate = init_lr,
        decay_rate = None,
        beta1 = 0.0,
        clipping_threshold = 1.0,
        factored = True,
        epsilon1 = 1e-30,
        epsilon2 = 0.001,
    )

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step = global_step
    )

    return train_op
