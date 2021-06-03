import tensorflow as tf
import numpy as np


def to_tf(input_nodes, inputs):
    if len(input_nodes) != len(inputs):
        raise Exception(
            'length of `input_nodes` not same as length of `inputs` for this eager graph.'
        )

    mapping = {}
    for i in range(len(input_nodes)):
        t = tf.convert_to_tensor(inputs[i])
        t = tf.cast(t, input_nodes[i].dtype)
        mapping[input_nodes[i].name.split(':')[0]] = t
    return mapping


def execute_graph(
    inputs,
    input_labels,
    output_labels,
    sess=None,
    input_nodes=None,
    output_nodes=None,
):
    output_nodes = {label: output_nodes[label] for label in output_labels}
    input_nodes = {
        input_nodes[label]: inputs[no] for no, label in enumerate(input_labels)
    }
    r = sess.run(output_nodes, feed_dict=input_nodes)

    return r
