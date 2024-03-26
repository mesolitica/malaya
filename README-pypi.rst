**Malaya** is a Natural-Language-Toolkit library for bahasa Malaysia, powered by PyTorch.

Documentation
--------------

Proper documentation is available at https://malaya.readthedocs.io/

Installing from the PyPI
----------------------------------

::

    $ pip install malaya

It will automatically install all dependencies except for PyTorch. So you can choose your own PyTorch CPU / GPU version.

Only **Python >= 3.6.0**,  and **PyTorch >= 1.10** are supported.

If you are a Windows user, make sure read https://malaya.readthedocs.io/en/latest/running-on-windows.html

Development Release
---------------------------------

Install from `master` branch,

::

    $ pip install git+https://github.com/huseinzol05/malaya.git


We recommend to use **virtualenv** for development. 

Documentation at https://malaya.readthedocs.io/en/latest/

Pretrained Models
------------------

Malaya also released Malaysian pretrained models, simply check at https://huggingface.co/mesolitica

References
-----------

If you use our software for research, please cite:

::

  @misc{Malaya, Natural-Language-Toolkit library for bahasa Malaysia, powered by PyTorch,
    author = {Husein, Zolkepli},
    title = {Malaya},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/mesolitica/malaya}}
  }

Acknowledgement
----------------

Thanks to `KeyReply <https://www.keyreply.com/>`_ for private V100s cloud and `Mesolitica <https://mesolitica.com/>`_ for private RTXs cloud to train Malaya-Speech models.

Also, thanks to `Tensorflow Research Cloud <https://www.tensorflow.org/tfrc>`_ for free TPUs access.

Contributing
----------------

Thank you for contributing this library, really helps a lot. Feel free to contact me to suggest me anything or want to contribute other kind of forms, we accept everything, not just code!
