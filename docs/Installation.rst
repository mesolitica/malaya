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

Installing for Windows / Anaconda
-----------------------------------

If you get error about `Microsoft Visual C++` during `pip` to install, you need to install Malaya without dependencies
::

      $ pip install sklearn xgboost==0.8
      $ pip install numpy scipy
      $ pip install tensorflow
      $ pip install pysastrawi
      $ pip install sklearn_crfsuite
      $ pip install fuzzywuzzy
      $ pip install requests
      $ pip install tqdm
      $ pip install unidecode
      $ pip install toolz
      $ pip install malaya --no-deps -U

But, if you follow these steps, you cannot use Word-Mover interface and fuzzywuzzy might be slow.

Dependencies
~~~~~~~~~~~~

Malaya depends on numpy, scipy, sklearn, tensorflow, xgboost, nltk, fuzzywuzzy, tqdm and toolz. Dependencies will install automatically during PIP.

Malaya depends on scikit-learn 0.19.1, any upper versions not recommended, stated by sklearn itself. You can install latest scikit-learn as you want, but it will may cause errors or models inefficiency.

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

GPU Environment
~~~~~~~~~~~~~~~

Deep learning Malaya trained on CUDA 9.0 and Tensorflow 1.12, supposedly any new version of CUDA and Tensorflow able to support Tensorflow 1.5 features.
