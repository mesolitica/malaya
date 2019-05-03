
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.3 s, sys: 1.59 s, total: 13.9 s
    Wall time: 18.3 s


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
6. Remove postfix for a word, ``kerjakan`` become ``kerja``.
7. Normalize elongated words, but this required speller object.

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
       remove_prefix_postfix = True,
       maxlen_segmenter = 20,
       validate = True,
       speller = None,
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

    string_1 = 'CANT WAIT for the new season of #mahathirmohamad ＼(^o^)／!!! #davidlynch #tvseries :))), TAAAK SAAABAAR!!!'
    string_2 = 'kecewa #johndoe movie and it suuuuucks!!! WASTED RM10... #badmovies :/'
    string_3 = "@husein:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    string_4 = 'aahhh, malasnye nak pegi keje harini #mondayblues'
    string_5 = '#drmahathir #najibrazak #1malaysia #mahathirnajib'

.. code:: ipython3

    %%time
    preprocessing = malaya.preprocessing.preprocessing()


.. parsed-literal::

    CPU times: user 14.5 s, sys: 1.54 s, total: 16 s
    Wall time: 16.7 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 8.35 ms, sys: 288 µs, total: 8.63 ms
    Wall time: 9.05 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 2.4 ms, sys: 70 µs, total: 2.47 ms
    Wall time: 2.58 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> <number> . <repeated> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.03 ms, sys: 10 µs, total: 1.04 ms
    Wall time: 1.13 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 2.06 ms, sys: 36 µs, total: 2.1 ms
    Wall time: 2.46 ms




.. parsed-literal::

    'aahh <elongated> , malas nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 60.3 ms, sys: 974 µs, total: 61.3 ms
    Wall time: 61.4 ms




.. parsed-literal::

    '<hashtag> dr mahathir </hashtag> <hashtag> najib razak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathir najib </hashtag>'



Load default paramaters with spelling correction to normalize elongated words.
------------------------------------------------------------------------------

We saw ``taak``, ``saabaar`` and another elongated words are not the
original words, so we can use spelling correction to normalize it.

.. code:: ipython3

    malays = malaya.load_malay_dictionary()
    corrector = malaya.spell.naive(malays)


.. parsed-literal::

    downloading Malay texts


.. parsed-literal::

    1.00MB [00:00, 9.83MB/s]                   


.. code:: ipython3

    %%time
    preprocessing = malaya.preprocessing.preprocessing(speller = corrector)


.. parsed-literal::

    CPU times: user 15.2 s, sys: 2.43 s, total: 17.6 s
    Wall time: 19 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 516 ms, sys: 14 ms, total: 530 ms
    Wall time: 533 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> talak <elongated> sabar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 92.6 ms, sys: 3.29 ms, total: 95.9 ms
    Wall time: 94.9 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> <number> . <repeated> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 149 ms, sys: 4.54 ms, total: 153 ms
    Wall time: 155 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> ya <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 515 ms, sys: 6.91 ms, total: 522 ms
    Wall time: 535 ms




.. parsed-literal::

    'amah <elongated> , malas nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 5.2 ms, sys: 327 µs, total: 5.53 ms
    Wall time: 5.59 ms




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

    CPU times: user 170 ms, sys: 35.4 ms, total: 206 ms
    Wall time: 220 ms


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 1.37 ms, sys: 54 µs, total: 1.42 ms
    Wall time: 1.49 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathirmohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> davidlynch </hashtag> <hashtag> tvseries </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 620 µs, sys: 38 µs, total: 658 µs
    Wall time: 672 µs




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

    CPU times: user 17.3 s, sys: 3.52 s, total: 20.9 s
    Wall time: 27.9 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 69.6 ms, sys: 1.52 ms, total: 71.1 ms
    Wall time: 72 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 2.2 ms, sys: 94 µs, total: 2.3 ms
    Wall time: 2.31 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> movie and it suucks <elongated> ! <repeated> <allcaps> wasted </allcaps> <money> <number> . <repeated> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.03 ms, sys: 44 µs, total: 1.08 ms
    Wall time: 1.09 ms




.. parsed-literal::

    '<user> : can not wait untuk the <date> <hashtag> sentiment </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'


