.. code:: python

    %%time
    
    import malaya


.. parsed-literal::

    CPU times: user 5.14 s, sys: 1.96 s, total: 7.1 s
    Wall time: 4.42 s


Check Malaya is GPU
-------------------

Make sure install ``malaya-gpu`` to utilize full gpu functions in
Malaya,

.. code:: bash


   pip install malaya-gpu

.. code:: python

    malaya.check_malaya_gpu()




.. parsed-literal::

    True



List available GPU
------------------

.. code:: python

    malaya.__gpu__




.. parsed-literal::

    ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']



.. code:: python

    !nvidia-smi


.. parsed-literal::

    Sun Jul 12 19:25:50 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.129      Driver Version: 410.129      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-DGXS...  On   | 00000000:07:00.0  On |                    0 |
    | N/A   44C    P0    39W / 300W |      0MiB / 32475MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
    | N/A   45C    P0    39W / 300W |      0MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
    | N/A   44C    P0    38W / 300W |      0MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   45C    P0    43W / 300W |      0MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


Right now all the GPUs in resting mode, no computation happened.

Limit GPU memory
----------------

By default Malaya will not set max cap for GPU memory, to put a cap,
override ``gpu_limit`` parameter in any load model API. ``gpu_limit``
should 0 < ``gpu_limit`` < 1. If ``gpu_limit = 0.3``, it means the model
will not use more than 30% of GPU memory.

.. code:: python


   malaya.sentiment.transformer(gpu_limit = 0.3)

N Models to N gpus
------------------

To allocate a model to another GPU, use ``gpu`` parameter, default is 0.

.. code:: python


   model_sentiment = malaya.sentiment.transformer(model = 'bert', gpu_limit = 0.5, gpu = 0)
   model_subjectivity = malaya.subjectivity.transformer(model = 'bert', gpu_limit = 0.5, gpu = 1)
   model_emotion = malaya.emotion.transformer(model = 'bert', gpu_limit = 0.5, gpu = 2)
   model_translation = malaya.translation.ms_en.transformer(gpu_limit = 0.5, gpu = 3)

GPU Rules
---------

1. Malaya will not consumed all available GPU memory, but slowly grow
   based on batch size. This growth only towards positive (use more GPU
   memory) dynamically, but will not reduce GPU memory if feed small
   batch size.
2. Use ``malaya.clear_session`` to clear session from unused models but
   this will not free GPU memory.
3. Even if you installed Malaya CPU version, it will always to load the
   models in GPU 0 first, if failed, it will load it in CPU.

.. code:: python

    anger_text = 'babi la company ni, aku dah la penat datang dari jauh'
    fear_text = 'takut doh tengok cerita hantu tadi'
    happy_text = 'bestnya dapat tidur harini, tak payah pergi kerja'
    love_text = 'aku sayang sgt dia dah doh'
    sadness_text = 'kecewa tengok kerajaan baru ni, janji ape pun tak dapat'
    surprise_text = 'sakit jantung aku, terkejut dengan cerita hantu tadi'

.. code:: python

    model_sentiment = malaya.sentiment.transformer(model = 'bert', gpu_limit = 0.5, gpu = 0)
    model_subjectivity = malaya.subjectivity.transformer(model = 'bert', gpu_limit = 0.5, gpu = 1)
    model_emotion = malaya.emotion.transformer(model = 'bert', gpu_limit = 0.5, gpu = 2)
    model_translation = malaya.translation.ms_en.transformer(gpu_limit = 0.5, gpu = 3)


.. parsed-literal::

    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:73: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:75: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:50: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:65: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: python

    %%time
    
    model_sentiment.predict_proba(
        [anger_text, fear_text, happy_text, love_text, sadness_text, surprise_text]
    )
    model_subjectivity.predict_proba(
        [anger_text, fear_text, happy_text, love_text, sadness_text, surprise_text]
    )
    model_emotion.predict_proba(
        [anger_text, fear_text, happy_text, love_text, sadness_text, surprise_text]
    )
    model_translation.translate(['Mahathir buat keputusan terburu-buru'])


.. parsed-literal::

    CPU times: user 8.61 s, sys: 2.71 s, total: 11.3 s
    Wall time: 10.8 s




.. parsed-literal::

    ['Mahathir made a hasty decision']



.. code:: python

    !nvidia-smi


.. parsed-literal::

    Sun Jul 12 19:26:18 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.129      Driver Version: 410.129      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-DGXS...  On   | 00000000:07:00.0  On |                    0 |
    | N/A   45C    P0    54W / 300W |   1101MiB / 32475MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
    | N/A   46C    P0    52W / 300W |   1100MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
    | N/A   45C    P0    52W / 300W |   1100MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   45C    P0    53W / 300W |   1100MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0     12786      C   /usr/bin/python3                            1089MiB |
    |    1     12786      C   /usr/bin/python3                            1089MiB |
    |    2     12786      C   /usr/bin/python3                            1089MiB |
    |    3     12786      C   /usr/bin/python3                            1089MiB |
    +-----------------------------------------------------------------------------+

