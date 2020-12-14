Speech Toolkit
================

.. raw:: html

    <p align="center">
        <a href="#readme">
            <img alt="logo" width="50%" src="https://malaya-dataset.s3-ap-southeast-1.amazonaws.com/malaya-speech.png">
        </a>
    </p>
    <p align="center">
        <a href="https://pypi.python.org/pypi/malaya-speech"><img alt="Pypi version" src="https://badge.fury.io/py/malaya-speech.svg"></a>
        <a href="https://pypi.python.org/pypi/malaya-speech"><img alt="Python3 version" src="https://img.shields.io/pypi/pyversions/malaya-speech.svg"></a>
        <a href="https://github.com/huseinzol05/Malaya-Speech/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/github/license/huseinzol05/malaya-speech.svg?color=blue"></a>
    </p>

=========

**Malaya-Speech** is a Speech-Toolkit library for bahasa Malaysia, powered by Deep Learning Tensorflow, we maintain it at separate repository, https://github.com/huseinzol05/malaya-speech

Documentation
--------------

Proper documentation is available at https://malaya-speech.readthedocs.io/

Installing from the PyPI
----------------------------------

CPU version
::

    $ pip install malaya-speech

GPU version
::

    $ pip install malaya-speech-gpu

Only **Python 3.6.x and above** and **Tensorflow 1.10 and above but not 2.0** are supported.

Features
--------
-  **Age Detection**

   Detect age in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker Diarization**

   Diarizing speakers using Pretrained Speaker Vector Malaya-Speech models.
-  **Emotion Detection**

   Detect emotions in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Gender Detection**

   Detect genders in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Language Detection**

   Detect hyperlocal languages in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Noise Reduction**

   Reduce multilevel noises using Pretrained STFT UNET Malaya-Speech models.
-  **Speaker Change**

   Detect changing speakers using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker overlap**

   Detect overlap speakers using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker Vector**

   Calculate similarity between speakers using Pretrained Malaya-Speech models.
-  **Speech Enhancement**

   Enhance voice activities using Pretrained STFT UNET Malaya-Speech models.
-  **Speech-to-Text**

   End-to-End Speech to Text using Pretrained CTC and RNN Transducer Malaya-Speech models.
-  **Vocoder**

   Convert Mel to Waveform using Pretrained MelGAN and HifiGAN Vocoder Malaya-Speech models.
-  **Voice Activity Detection**

   Detect voice activities using Finetuned Speaker Vector Malaya-Speech models.

Acknowledgement
----------------

Thanks to `KeyReply <https://www.keyreply.com/>`_ for sponsoring private cloud to train Malaya-Speech models, without it, this library will collapse entirely.  

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://cdn.techinasia.com/data/images/16234a59ae3f218dc03815a08eaab483.png">
    </a>