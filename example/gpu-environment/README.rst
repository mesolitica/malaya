One model always consumed one unit of gpu. For now we do not support
distributed batch processing to multiple GPUs from one model. But we can
initiate multiple models to multiple GPUs.

model_emotion -> GPU0

model_sentiment -> GPU1

model_translation -> GPU2

and so on.

.. code:: ipython3

    !git pull

.. code:: ipython3

    %%time
    
    import malaya


.. parsed-literal::

    CPU times: user 5.94 s, sys: 2.35 s, total: 8.29 s
    Wall time: 4.01 s


.. code:: ipython3

    !nvidia-smi


.. parsed-literal::

    Tue Jul  7 21:32:37 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.129      Driver Version: 410.129      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-DGXS...  On   | 00000000:07:00.0  On |                    0 |
    | N/A   55C    P0    42W / 300W |      0MiB / 32475MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
    | N/A   65C    P0   250W / 300W |  31452MiB / 32478MiB |     89%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
    | N/A   63C    P0   270W / 300W |  31452MiB / 32478MiB |     92%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   63C    P0   252W / 300W |  31452MiB / 32478MiB |     77%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    1     11646      C   python3                                    31431MiB |
    |    2     11646      C   python3                                    31431MiB |
    |    3     11646      C   python3                                    31431MiB |
    +-----------------------------------------------------------------------------+


Right now all the GPUs in resting mode, no computation happened.

GPU Rules
---------

1. By default, all models will initiate in first GPU, unless override
   ``gpu`` parameter in any load model API. Example as below.
2. Malaya will not consumed all available GPU memory, but slowly grow
   based on batch size. This growth only towards positive (use more GPU
   memory) dynamically, but will not reduce GPU memory if feed small
   batch size.
3. Use ``malaya.clear_session`` to clear session from unused models but
   this will not free GPU memory.
4. By default Malaya will not set max cap for GPU memory, to put a cap,
   override ``gpu_limit`` parameter in any load model API. ``gpu_limit``
   should 0 < ``gpu_limit`` < 1. If ``gpu_limit = 0.3``, it means the
   model will not use more than 30% of GPU memory.

.. code:: ipython3

    anger_text = 'babi la company ni, aku dah la penat datang dari jauh'
    fear_text = 'takut doh tengok cerita hantu tadi'
    happy_text = 'bestnya dapat tidur harini, tak payah pergi kerja'
    love_text = 'aku sayang sgt dia dah doh'
    sadness_text = 'kecewa tengok kerajaan baru ni, janji ape pun tak dapat'
    surprise_text = 'sakit jantung aku, terkejut dengan cerita hantu tadi'

.. code:: ipython3

    model = malaya.emotion.transformer(model = 'bert', gpu = '0')


.. parsed-literal::

    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:61: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:62: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:50: The name tf.GPUOptions is deprecated. Please use tf.compat.v1.GPUOptions instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:51: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:53: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    %%time
    
    model.predict_proba(
        [anger_text, fear_text, happy_text, love_text, sadness_text, surprise_text]
    )


.. parsed-literal::

    CPU times: user 1.94 s, sys: 541 ms, total: 2.48 s
    Wall time: 2.52 s




.. parsed-literal::

    [{'anger': 0.9998965,
      'fear': 1.7692768e-05,
      'happy': 1.8747674e-05,
      'love': 1.656881e-05,
      'sadness': 3.130815e-05,
      'surprise': 1.9183277e-05},
     {'anger': 7.4469484e-05,
      'fear': 0.99977416,
      'happy': 6.824215e-05,
      'love': 2.773282e-05,
      'sadness': 1.9767067e-05,
      'surprise': 3.5663204e-05},
     {'anger': 0.99963737,
      'fear': 3.931449e-05,
      'happy': 0.0001562279,
      'love': 3.3580774e-05,
      'sadness': 0.00011328616,
      'surprise': 2.0134145e-05},
     {'anger': 3.1319763e-05,
      'fear': 1.7286226e-05,
      'happy': 2.9899325e-05,
      'love': 0.99987257,
      'sadness': 2.7867774e-05,
      'surprise': 2.096328e-05},
     {'anger': 8.965934e-05,
      'fear': 1.8196944e-05,
      'happy': 2.9275663e-05,
      'love': 1.7211949e-05,
      'sadness': 0.9998247,
      'surprise': 2.0944033e-05},
     {'anger': 4.132152e-05,
      'fear': 6.202527e-05,
      'happy': 3.1012056e-05,
      'love': 5.3896296e-05,
      'sadness': 6.202101e-05,
      'surprise': 0.9997497}]



.. code:: ipython3

    !nvidia-smi


.. parsed-literal::

    Tue Jul  7 21:32:57 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.129      Driver Version: 410.129      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-DGXS...  On   | 00000000:07:00.0  On |                    0 |
    | N/A   56C    P0    58W / 300W |   1099MiB / 32475MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
    | N/A   64C    P0   219W / 300W |  31452MiB / 32478MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
    | N/A   62C    P0   248W / 300W |  31452MiB / 32478MiB |     99%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   62C    P0   236W / 300W |  31452MiB / 32478MiB |     76%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0      2536      C   /usr/bin/python3                            1087MiB |
    |    1     11646      C   python3                                    31431MiB |
    |    2     11646      C   python3                                    31431MiB |
    |    3     11646      C   python3                                    31431MiB |
    +-----------------------------------------------------------------------------+


.. code:: ipython3

    malaya.clear_session(model)




.. parsed-literal::

    True



