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

Dependencies
~~~~~~~~~~~~

Malaya depends on numpy, scipy, sklearn, tensorflow, xgboost, nltk, fuzzywuzzy, tqdm and toolz. Dependencies will install automatically during PIP.

Malaya depends on scikit-learn 0.19.1, any upper versions not recommended, stated by sklearn itself. You can install latest scikit-learn as you want, but it will may cause errors or models inefficiency.

From Source
-----------

Malaya is actively developed on
`Github <https://github.com/devconx/malaya>`__.

You can clone the public repo:

.. code:: python

    git clone https://github.com/devconx/malaya

Once you have the source, you can install it into your site-packages
with:

.. code:: python

    python setup.py install

Python
~~~~~~

Malaya trained on Python 3.6, supposedly able to support Python 3.X but below than Python 3.7. There is no released version Tensorflow for Python 3.7.

GPU Environment
~~~~~~~~~~~~~~~

Deep learning Malaya trained on CUDA 8.0 and Tensorflow 1.5, supposedly any new version of CUDA and Tensorflow able to support Tensorflow 1.5 features.
