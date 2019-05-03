
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.1 s, sys: 1.73 s, total: 13.8 s
    Wall time: 19.1 s


Preprocessing
-------------

We know that social media texts from Twitter, Facebook and Instagram are
very noisy and we want to clean as much as possible to make our machines
understand the structure of sentence much better. In Malaya, we
standardize our text preprocessing,

1. Malaya can replace special words into tokens to reduce dimension
   curse. ``rm10`` become ``<money> <number>``.
2. Malaya can expand hashtags, ``#drmahathir`` become ``dr mahathir``.
3. Malaya can put tags for special words, ``#drmahathir`` become
   ``<hashtag> drmathir </hashtag>``.
4. Malaya can expand english contractions.
5. Malaya can translate english words to become bahasa malaysia words.
   Again, this translation is using dictionary, it will not understand
   semantically. Purpose of this translation just to standardize to
   become bahasa Malaysia.

These are defaults setting for ``preprocessing()``,

::

   def preprocessing(
       normalize = [
           'url',
           'email',
           'percent',
           'money',
           'phone',
           'user',
           'time',
           'date',
           'number',
       ],
       annotate = [
           'allcaps',
           'elongated',
           'repeated',
           'emphasis',
           'censored',
           'hashtag',
       ],
       lowercase = True,
       fix_unidecode = True,
       expand_hashtags = True,
       expand_english_contractions = True,
       translate_english_to_bm = True,
       maxlen_segmenter = 20,
       validate = True,
   ):

normalize
^^^^^^^^^

Supported ``normalize``,

1.  hashtag
2.  cashtag
3.  tag
4.  user
5.  emphasis
6.  censored
7.  acronym
8.  eastern_emoticons
9.  rest_emoticons
10. emoji
11. quotes
12. percent
13. repeat_puncts
14. money
15. email
16. phone
17. number
18. allcaps
19. url
20. date
21. time

You can check all supported list at
``malaya.preprocessing.get_normalize()``.

Example, if you set ``money`` and ``number``, and input string is
``RM10``, the output is ``<money> <number>``.

annotate
^^^^^^^^

Supported ``annotate``,

1. hashtag
2. allcaps
3. elongated
4. repeated
5. emphasis
6. censored

Example, if you set ``hashtag``, and input string is ``#drmahathir``,
the output is ``<hashtag> drmahathir </hashtag>``.

Load default paramaters
-----------------------

default parameters able to expand hashtag, ``#mahathirmohamad`` into
``mahathir mohamad``, but initial load is quite slow and translate
english to bahasa malaysia.

.. code:: ipython3

    %%time
    preprocessing = malaya.preprocessing.preprocessing()


.. parsed-literal::

    downloading frozen /Users/huseinzol/Malaya/preprocessing model


.. parsed-literal::

    6.00MB [00:02, 2.28MB/s]                          


.. parsed-literal::

    CPU times: user 14.8 s, sys: 2.2 s, total: 17 s
    Wall time: 20.8 s


.. code:: ipython3

    string_1 = 'CANT WAIT for the new season of #mahathirmohamad ＼(^o^)／!!! #davidlynch #tvseries :))), TAAAK SAAABAAR!!!'
    string_2 = 'kecewa #johndoe movie and it suuuuucks!!! WASTED RM10... #badmovies :/'
    string_3 = "@husein:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    string_4 = 'aahhh, malasnye nak pegi keje harini #mondayblues'
    string_5 = '#drmahathir #najibrazak #1malaysia #mahathirnajib'

.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 10.3 ms, sys: 1.51 ms, total: 11.9 ms
    Wall time: 11.8 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 1.8 ms, sys: 98 µs, total: 1.9 ms
    Wall time: 1.91 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazirkan </allcaps> <money> <number> . <repeated> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 748 µs, sys: 17 µs, total: 765 µs
    Wall time: 774 µs




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 2.19 ms, sys: 108 µs, total: 2.29 ms
    Wall time: 2.35 ms




.. parsed-literal::

    'aahh <elongated> , malasnye nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 8.86 ms, sys: 1.11 ms, total: 9.97 ms
    Wall time: 9.2 ms




.. parsed-literal::

    '<hashtag> dr mahathir </hashtag> <hashtag> najib razak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathir najib </hashtag>'



disable hashtag expander
------------------------

Sometime we want to maintain hashtags because we want to study the
frequencies.

.. code:: ipython3

    %%time
    preprocessing = malaya.preprocessing.preprocessing(expand_hashtags = False)


.. parsed-literal::

    CPU times: user 155 ms, sys: 30.8 ms, total: 186 ms
    Wall time: 190 ms


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 1.05 ms, sys: 15 µs, total: 1.07 ms
    Wall time: 1.08 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathirmohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> davidlynch </hashtag> <hashtag> tvseries </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 633 µs, sys: 1 µs, total: 634 µs
    Wall time: 645 µs




.. parsed-literal::

    '<hashtag> drmahathir </hashtag> <hashtag> najibrazak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathirnajib </hashtag>'



disable english translation
---------------------------

But there are basic normalizations that cannot override, like, ``for``
automatically become ``untuk``. You can check default entire
normalizations at
``from malaya.texts._tatabahasa import rules_normalizer``

.. code:: ipython3

    %%time
    preprocessing = malaya.preprocessing.preprocessing(translate_english_to_bm = False)


.. parsed-literal::

    CPU times: user 15.3 s, sys: 3 s, total: 18.3 s
    Wall time: 21 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 76.2 ms, sys: 4.24 ms, total: 80.5 ms
    Wall time: 80.5 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 134 ms, sys: 3.88 ms, total: 137 ms
    Wall time: 138 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> movie and it suucks <elongated> ! <repeated> <allcaps> wasted </allcaps> <money> <number> . <repeated> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 703 µs, sys: 18 µs, total: 721 µs
    Wall time: 729 µs




.. parsed-literal::

    '<user> : can not wait untuk the <date> <hashtag> sentiment </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'


