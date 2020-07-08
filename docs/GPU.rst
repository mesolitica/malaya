GPU Environment
================

Deep learning Malaya trained on CUDA 10.0 and Tensorflow 1.13, supposedly any new version of CUDA and Tensorflow able to support Tensorflow >= 1.13 features.

To prevent any CPU and GPU version conflict, should try to uninstall all Malaya version first,
::

    $ pip uninstall malaya malaya-gpu

After that simply install gpu version,
::

    $ pip install malaya-gpu

GPU Version Benefit
--------------------

1. Different models different GPUs.
2. Automatically try to use cugraph for any networkx functions.

Different models different GPUs
----------------------------------

.. note::

    This tutorial is available as an IPython notebook
    `here <https://github.com/huseinzol05/Malaya/tree/master/example/gpu-environment>`_.

.. include:: gpu-environment.rst