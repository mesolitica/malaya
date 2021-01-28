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

**Malaya-Speech** is a Speech-Toolkit library for bahasa Malaysia, powered by Deep Learning Tensorflow. We maintain it at separate repository, https://github.com/huseinzol05/malaya-speech

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

-  **Age Detection**, detect age in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker Diarization**, diarizing speakers using Pretrained Speaker Vector Malaya-Speech models.
-  **Emotion Detection**, detect emotions in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Gender Detection**, detect genders in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Language Detection**, detect hyperlocal languages in speech using Finetuned Speaker Vector Malaya-Speech models.
-  **Noise Reduction**, reduce multilevel noises using Pretrained STFT UNET Malaya-Speech models.
-  **Speaker Change**, detect changing speakers using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker overlap**, detect overlap speakers using Finetuned Speaker Vector Malaya-Speech models.
-  **Speaker Vector**, calculate similarity between speakers using Pretrained Malaya-Speech models.
-  **Speech Enhancement**, enhance voice activities using Pretrained Waveform UNET Malaya-Speech models.
-  **Speech-to-Text**, End-to-End Speech to Text using Pretrained CTC and RNN Transducer Malaya-Speech models.
-  **Super Resolution**, Super Resolution 4x using Pretrained Super Resolution Malaya-Speech models.
-  **Text-to-Speech**, using Pretrained Tacotron2 and FastSpeech2 Malaya-Speech models.
-  **Vocoder**, convert Mel to Waveform using Pretrained MelGAN, Multiband MelGAN and Universal MelGAN Vocoder Malaya-Speech models.
-  **Voice Activity Detection**, detect voice activities using Finetuned Speaker Vector Malaya-Speech models.

Pretrained Models
------------------

Malaya-Speech also released pretrained models, simply check at `malaya-speech/pretrained-model <https://github.com/huseinzol05/malaya-speech/tree/master/pretrained-model>`_

-  **Wave UNET**,  Multi-Scale Neural Network for End-to-End Audio Source Separation, https://arxiv.org/abs/1806.03185
-  **Wave ResNet UNET**, added ResNet style into Wave UNET, no paper produced.
-  **Deep Speaker**, An End-to-End Neural Speaker Embedding System, https://arxiv.org/pdf/1705.02304.pdf
-  **SpeakerNet**, 1D Depth-wise Separable Convolutional Network for Text-Independent Speaker Recognition and Verification, https://arxiv.org/abs/2010.12653
-  **VGGVox**, a large-scale speaker identification dataset, https://arxiv.org/pdf/1706.08612.pdf
-  **GhostVLAD**, Utterance-level Aggregation For Speaker Recognition In The Wild, https://arxiv.org/abs/1902.10107
-  **Conformer**, Convolution-augmented Transformer for Speech Recognition, https://arxiv.org/abs/2005.08100
-  **ALConformer**, A lite Conformer, no paper produced.
-  **Jasper**, An End-to-End Convolutional Neural Acoustic Model, https://arxiv.org/abs/1904.03288
-  **Tacotron2**, Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions, https://arxiv.org/abs/1712.05884
-  **FastSpeech2**, Fast and High-Quality End-to-End Text to Speech, https://arxiv.org/abs/2006.04558
-  **MelGAN**, Generative Adversarial Networks for Conditional Waveform Synthesis, https://arxiv.org/abs/1910.06711
-  **Multi-band MelGAN**, Faster Waveform Generation for High-Quality Text-to-Speech, https://arxiv.org/abs/2005.05106
-  **SRGAN**, Modified version of SRGAN to do 1D Convolution, Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, https://arxiv.org/abs/1609.04802
-  **Speech Enhancement UNET**, https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement
-  **Universal MelGAN**, Universal MelGAN: A Robust Neural Vocoder for High-Fidelity Waveform Generation in Multiple Domains, https://arxiv.org/abs/2011.09631

References
-----------

If you use our software for research, please cite:

::

  @misc{Malaya, Speech-Toolkit library for bahasa Malaysia, powered by Deep Learning Tensorflow,
    author = {Husein, Zolkepli},
    title = {Malaya-Speech},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/huseinzol05/malaya-speech}}
  }

Acknowledgement
----------------

Thanks to `KeyReply <https://www.keyreply.com/>`_ for sponsoring private cloud to train Malaya-Speech models, without it, this library will collapse entirely.  

.. raw:: html

    <a href="#readme">
        <img alt="logo" width="20%" src="https://cdn.techinasia.com/data/images/16234a59ae3f218dc03815a08eaab483.png">
    </a>