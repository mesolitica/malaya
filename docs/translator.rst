We use a translator to translate from a validated English dataset to a
Bahasa dataset.

Everyone agree that Google Translate is the best online translator in
this world, but the problem here, to subscribe the API from Google Cloud
is really insane expensive.

Good thing about https://translate.google.com/, it open for public
internet! So we just code a headless browser using Selenium with
PhantomJS as the backbone, that’s all!

You can check the source code here,
`translator/ <https://github.com/huseinzol05/Malaya/tree/master/translator>`__

.. code:: python

    from translate_selenium import Translate, Translate_Concurrent

Translate a sentence
--------------------

.. code:: python

    with open('sample-joy') as fopen:
        dataset = list(filter(None, fopen.read().split('\n')))
    len(dataset)




.. parsed-literal::

    18



.. code:: python

    translator = Translate(from_lang = 'en', to_lang = 'ms')

You can get list of supported language in here,
https://cloud.google.com/translate/docs/languages

.. code:: python

    %%time
    translator.translate(dataset[0])


.. parsed-literal::

    CPU times: user 4 ms, sys: 0 ns, total: 4 ms
    Wall time: 1.23 s




.. parsed-literal::

    'seorang lelaki yang saya mengagumi begitu banyak meminta saya untuk pergi bersamanya'



1.23 seconds, it took a very long time to translate a single sentence.
What if you have 100k of sentences? It will cost you around 123000
seconds! insane to wait!

So, we provide multihreading translator, concurrently translate multi
sentences.

Translate batch of strings
--------------------------

.. code:: python

    translators = Translate_Concurrent(batch_size = 3, from_lang = 'en', to_lang = 'ms')

.. code:: python

    %%time
    translators.translate_batch(dataset[:3])


.. parsed-literal::

    100%|███████████████████████████████████| 1/1 [00:01<00:00,  1.44s/it]

.. parsed-literal::

    CPU times: user 8 ms, sys: 12 ms, total: 20 ms
    Wall time: 1.44 s


.. parsed-literal::






.. parsed-literal::

    ['kawan yang sudah berkahwin rapat hanya mempunyai anak pertamanya',
     'pengenalan rapat menangis untuk saya saya merasa gembira kerana ada yang peduli',
     'seorang lelaki yang saya mengagumi begitu banyak meminta saya untuk pergi bersamanya']



See, we predicted 3 sentences at almost wall time. You can increase the
``batch_size`` to any size you want, limit is your spec now, this method
will never make Google blocked your IP. Malaya already tested it more
than 300k of sentences.

Remember, 1 translator took a quite toll, here I spawned 10 translators,
look from my ``top``,

.. code:: text

   PID   USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
   14628 husein    20   0 3175700 398980  43036 S  33.6  2.4   5:38.05 phantomjs
   14652 husein    20   0 3188824 408880  43084 S  29.9  2.5   5:34.62 phantomjs
   14489 husein    20   0 3204708 411520  43064 S  28.6  2.5   5:35.29 phantomjs
   14466 husein    20   0 3171668 400304  43008 S  24.6  2.5   5:26.74 phantomjs
   14443 husein    20   0 3181056 403228  42916 S  21.9  2.5   5:26.24 phantomjs
   14512 husein    20   0 3187592 416036  42956 S  20.3  2.6   5:30.03 phantomjs
   14558 husein    20   0 3206104 419800  43640 S  19.9  2.6   5:30.76 phantomjs
   14535 husein    20   0 3179416 405508  43196 S  18.3  2.5   5:27.54 phantomjs
   14420 husein    20   0 3202472 422448  43064 S  17.6  2.6   5:26.78 phantomjs
   14581 husein    20   0 3181132 401892  43056 S  16.3  2.5   5:33.48 phantomjs

1 translator cost me around,

.. code:: text

   PID   USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
   14628 husein    20   0 3175700 398980  43036 S  33.6  2.4   5:38.05 phantomjs

My machine specifications,

.. code:: text

   H/W path       Device       Class          Description
   ======================================================
                               system         G1.Sniper H6 (To be filled by O.E.M.)
   /0                          bus            G1.Sniper H6
   /0/3d                       processor      Intel(R) Core(TM) i5-4690 CPU @ 3.50GHz
   /0/42                       memory         16GiB System Memory
   /0/42/0                     memory         DIMM [empty]
   /0/42/1                     memory         8GiB DIMM DDR3 Synchronous 1600 MHz (0.6 ns)
   /0/42/2                     memory         DIMM [empty]
   /0/42/3                     memory         8GiB DIMM DDR3 Synchronous 1600 MHz (0.6 ns)
   /0/100                      bridge         4th Gen Core Processor DRAM Controller
   /0/100/1                    bridge         Xeon E3-1200 v3/4th Gen Core Processor PCI Express x16 Controller
   /0/100/1/0                  display        GM206 [GeForce GTX 960]
   /0/100/1/0.1                multimedia     NVIDIA Corporation

**So, beware of your machine!**
