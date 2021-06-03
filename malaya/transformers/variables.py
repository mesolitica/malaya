# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Variable functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
import functools


def variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    device=None,
    partitioner=None,
    custom_getter=None,
    use_resource=None,
    synchronization=variables.VariableSynchronization.AUTO,
    aggregation=variables.VariableAggregation.NONE,
):
    """Gets an existing variable with these parameters or creates a new one.
  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of applying
      it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: A list of collection names to which the Variable will be added.
      If None it would default to `tf.GraphKeys.GLOBAL_VARIABLES`.
    caching_device: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and dtype of the `Variable` to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    custom_getter: Callable that allows overwriting the internal get_variable
      method and has to have the same signature.
    use_resource: If `True` use a ResourceVariable instead of a Variable.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableSynchronization`. By default the synchronization is set to
      `AUTO` and the current `DistributionStrategy` chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableAggregation`.
  Returns:
    The created or existing variable.
  """
    collections = list(
        collections
        if collections is not None
        else [ops.GraphKeys.GLOBAL_VARIABLES]
    )

    # Remove duplicates
    collections = list(set(collections))
    getter = variable_scope.get_variable
    if custom_getter is not None:
        getter = functools.partial(
            custom_getter, reuse=variable_scope.get_variable_scope().reuse
        )
    with ops.device(device or ''):
        return getter(
            name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            trainable=trainable,
            collections=collections,
            caching_device=caching_device,
            partitioner=partitioner,
            use_resource=use_resource,
            synchronization=synchronization,
            aggregation=aggregation,
        )


def model_variable(
    name,
    shape=None,
    dtype=dtypes.float32,
    initializer=None,
    regularizer=None,
    trainable=True,
    collections=None,
    caching_device=None,
    device=None,
    partitioner=None,
    custom_getter=None,
    use_resource=None,
    synchronization=variables.VariableSynchronization.AUTO,
    aggregation=variables.VariableAggregation.NONE,
):
    """Gets an existing model variable with these parameters or creates a new one.
  Args:
    name: the name of the new or existing variable.
    shape: shape of the new or existing variable.
    dtype: type of the new or existing variable (defaults to `DT_FLOAT`).
    initializer: initializer for the variable if one is created.
    regularizer: a (Tensor -> Tensor or None) function; the result of applying
      it on a newly created variable will be added to the collection
      GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    collections: A list of collection names to which the Variable will be added.
      Note that the variable is always also added to the
      `GraphKeys.GLOBAL_VARIABLES` and `GraphKeys.MODEL_VARIABLES` collections.
    caching_device: Optional device string or function describing where the
      Variable should be cached for reading.  Defaults to the Variable's device.
    device: Optional device to place the variable. It can be an string or a
      function that is called to get the device for the variable.
    partitioner: Optional callable that accepts a fully defined `TensorShape`
      and dtype of the `Variable` to be created, and returns a list of
      partitions for each axis (currently only one axis can be partitioned).
    custom_getter: Callable that allows overwriting the internal get_variable
      method and has to have the same signature.
    use_resource: If `True` use a ResourceVariable instead of a Variable.
    synchronization: Indicates when a distributed a variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableSynchronization`. By default the synchronization is set to
      `AUTO` and the current `DistributionStrategy` chooses when to synchronize.
    aggregation: Indicates how a distributed variable will be aggregated.
      Accepted values are constants defined in the class
      `tf.VariableAggregation`.
  Returns:
    The created or existing variable.
  """
    collections = list(collections or [])
    collections += [
        ops.GraphKeys.GLOBAL_VARIABLES,
        ops.GraphKeys.MODEL_VARIABLES,
    ]
    var = variable(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        collections=collections,
        caching_device=caching_device,
        device=device,
        partitioner=partitioner,
        custom_getter=custom_getter,
        use_resource=use_resource,
        synchronization=synchronization,
        aggregation=aggregation,
    )
    return var
