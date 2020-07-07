GPU Environment
================

Deep learning Malaya trained on CUDA 10.0 and Tensorflow 1.15, supposedly any new version of CUDA and Tensorflow able to support Tensorflow >= 1.13 features.

Simply install gpu version,
::

    $ pip install malaya-gpu


Or, manually install tensorflow-gpu,
::

    $ pip install tensorflow-gpu==1.15

If Tensorflow gpu found in the local, all Malaya models will automatically run in GPU.

Different models different GPUs
----------------------------------

