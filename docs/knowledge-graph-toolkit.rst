Knowledge Graph Toolkit
=======================

.. raw:: html

    <p align="center">
        <a href="#readme">
            <img alt="logo" width="40%" src="https://i.imgur.com/YaX8guI.png">
        </a>
    </p>
    <p align="center">
        <a href="https://github.com/huseinzol05/malaya-graph/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/huseinzol05/malaya-graph.svg?color=blue"></a>
        <a href="https://discord.gg/aNzbnRqt3A"><img alt="discord" src="https://img.shields.io/badge/discord%20server-malaya-rgb(118,138,212).svg"></a>
    </p>

=========

**Malaya-Graph** is a Knowledge Graph Toolkit for Bahasa Malaysia, powered by Tensorflow and PyTorch.

Documentation
--------------

Proper documentation is available at https://malaya-graph.readthedocs.io/

Installing from the PyPI
----------------------------------

::

    $ pip install malaya-graph

It will automatically install all dependencies except for Tensorflow and PyTorch. So you can choose your own Tensorflow CPU / GPU version and PyTorch CPU / GPU version.

Only **Python >= 3.6.0**, **Tensorflow >= 1.15.0**, and **PyTorch >= 1.10** are supported.

Development Release
---------------------------------

Install from `master` branch,

::

    $ pip install git+https://github.com/huseinzol05/malaya-graph.git


We recommend to use **virtualenv** for development. 

Documentation at https://malaya-graph.readthedocs.io/en/latest/

Features
--------

-  **Text to Knowledge Graph**, Generate knowledge graph from human sentences.
-  **Is Triplet**, Clasify a triplet is a correct head-type-tail relationship.
-  **Knowledge Graph to Text**, Generate human sentences from a knowledge graph.
-  **Link Prediction**, Zero-shot link classifier between a head and a tail.
-  **Relation Prediction**, Zero-shot head or tail classifier.

References
-----------

If you use our software for research, please cite:

::

  @misc{Malaya, Knowledge Graph Toolkit for Bahasa Malaysia, powered by Tensorflow and PyTorch,
    author = {Husein, Zolkepli},
    title = {Malaya-Graph},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/huseinzol05/malaya-graph}}
  }

Acknowledgement
----------------

Thanks to `Mesolitica <https://mesolitica.com/>`_ for private RTXs cloud to train Malaya-Graph models,

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://i1.wp.com/mesolitica.com/wp-content/uploads/2019/06/Mesolitica_Logo_Only.png?fit=857%2C532&ssl=1">
    </a>