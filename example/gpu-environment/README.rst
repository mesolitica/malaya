.. code:: ipython3

    %%time
    
    import malaya


.. parsed-literal::

    CPU times: user 5.79 s, sys: 2.45 s, total: 8.24 s
    Wall time: 3.63 s


.. code:: ipython3

    malaya.gpu_available()




.. parsed-literal::

    True



.. code:: ipython3

    !nvidia-smi


.. parsed-literal::

    Fri Jul 10 12:39:26 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.129      Driver Version: 410.129      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-DGXS...  On   | 00000000:07:00.0  On |                    0 |
    | N/A   43C    P0    39W / 300W |      0MiB / 32475MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
    | N/A   45C    P0    39W / 300W |      0MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
    | N/A   44C    P0    38W / 300W |      0MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   44C    P0    40W / 300W |      0MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+


Right now all the GPUs in resting mode, no computation happened.

GPU Rules
---------

1. Malaya will not consumed all available GPU memory, but slowly grow
   based on batch size. This growth only towards positive (use more GPU
   memory) dynamically, but will not reduce GPU memory if feed small
   batch size.
2. Use ``malaya.clear_session`` to clear session from unused models but
   this will not free GPU memory.
3. By default Malaya will not set max cap for GPU memory, to put a cap,
   override ``gpu_limit`` parameter in any load model API. ``gpu_limit``
   should 0 < ``gpu_limit`` < 1. If ``gpu_limit = 0.3``, it means the
   model will not use more than 30% of GPU memory.
4. Even if you installed Malaya CPU version, it will always to load the
   models in GPU first, if failed, it will load it in CPU.

.. code:: ipython3

    anger_text = 'babi la company ni, aku dah la penat datang dari jauh'
    fear_text = 'takut doh tengok cerita hantu tadi'
    happy_text = 'bestnya dapat tidur harini, tak payah pergi kerja'
    love_text = 'aku sayang sgt dia dah doh'
    sadness_text = 'kecewa tengok kerajaan baru ni, janji ape pun tak dapat'
    surprise_text = 'sakit jantung aku, terkejut dengan cerita hantu tadi'

.. code:: ipython3

    model = malaya.emotion.transformer(model = 'bert', gpu_limit = 0.5)


.. parsed-literal::

    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:72: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:73: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:58: The name tf.GPUOptions is deprecated. Please use tf.compat.v1.GPUOptions instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:61: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.
    
    WARNING:tensorflow:From /home/husein/malaya/Malaya/malaya/function/__init__.py:63: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    %%time
    
    model.predict_proba(
        [anger_text, fear_text, happy_text, love_text, sadness_text, surprise_text]
    )


.. parsed-literal::

    CPU times: user 1.8 s, sys: 504 ms, total: 2.3 s
    Wall time: 2.3 s




.. parsed-literal::

    [{'anger': 0.99989223,
      'fear': 1.5843118e-05,
      'happy': 1.660186e-05,
      'love': 1.9634477e-05,
      'sadness': 3.827092e-05,
      'surprise': 1.7427232e-05},
     {'anger': 4.894743e-05,
      'fear': 0.999795,
      'happy': 6.764499e-05,
      'love': 3.6289443e-05,
      'sadness': 1.9702624e-05,
      'surprise': 3.2430926e-05},
     {'anger': 0.9997905,
      'fear': 2.5795038e-05,
      'happy': 6.7572015e-05,
      'love': 2.6636817e-05,
      'sadness': 6.734582e-05,
      'surprise': 2.2285754e-05},
     {'anger': 2.4449551e-05,
      'fear': 2.6033362e-05,
      'happy': 3.1518703e-05,
      'love': 0.9998758,
      'sadness': 1.895303e-05,
      'surprise': 2.326243e-05},
     {'anger': 8.095824e-05,
      'fear': 2.3824483e-05,
      'happy': 2.1045413e-05,
      'love': 1.6150812e-05,
      'sadness': 0.99983835,
      'surprise': 1.9708685e-05},
     {'anger': 4.470948e-05,
      'fear': 0.00010641558,
      'happy': 2.9055469e-05,
      'love': 4.5270677e-05,
      'sadness': 5.7159534e-05,
      'surprise': 0.9997173}]



.. code:: ipython3

    !nvidia-smi


.. parsed-literal::

    Fri Jul 10 12:39:56 2020       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 410.129      Driver Version: 410.129      CUDA Version: 10.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  Tesla V100-DGXS...  On   | 00000000:07:00.0  On |                    0 |
    | N/A   44C    P0    54W / 300W |   1099MiB / 32475MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   1  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
    | N/A   45C    P0    52W / 300W |    418MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   2  Tesla V100-DGXS...  On   | 00000000:0E:00.0 Off |                    0 |
    | N/A   44C    P0    51W / 300W |    418MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
    |   3  Tesla V100-DGXS...  On   | 00000000:0F:00.0 Off |                    0 |
    | N/A   45C    P0    54W / 300W |    418MiB / 32478MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |    0     35310      C   /usr/bin/python3                            1087MiB |
    |    1     35310      C   /usr/bin/python3                             407MiB |
    |    2     35310      C   /usr/bin/python3                             407MiB |
    |    3     35310      C   /usr/bin/python3                             407MiB |
    +-----------------------------------------------------------------------------+


.. code:: ipython3

    malaya.clear_session(model)




.. parsed-literal::

    True



