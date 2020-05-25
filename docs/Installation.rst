Installation
============

Installing/Upgrading From the PyPI
----------------------------------

CPU version
::

    $ pip install malaya

GPU version
::

    $ pip install malaya-gpu

From Source
-----------

Malaya is actively developed on
`Github <https://github.com/huseinzol05/malaya>`__.

You can clone the public repo:

.. code:: python

    git clone https://github.com/huseinzol05/malaya

Once you have the source, you can install it into your site-packages
with:

.. code:: python

    python setup.py install

Python
~~~~~~

Malaya only supported Python 3.6 and above.

Tensorflow
~~~~~~~~~~~

Malaya only supported 1.14 <= Tensorflow < 2.0. 

We have no intention for now to upgrade Tensorflow to 2.0 because we depends a lot from contrib package Tensorflow 1.1X .

GPU Environment
~~~~~~~~~~~~~~~

Deep learning Malaya trained on CUDA 9.0 and Tensorflow 1.12, supposedly any new version of CUDA and Tensorflow able to support Tensorflow >= 1.13 features.
