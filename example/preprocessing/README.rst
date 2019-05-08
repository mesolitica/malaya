
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.8 s, sys: 2.05 s, total: 14.8 s
    Wall time: 22.2 s


Preprocessing
-------------

We know that social media texts from Twitter, Facebook and Instagram are
very noisy and we want to clean as much as possible to make our machines
understand the structure of sentence much better. In Malaya, we
standardize our text preprocessing,

1. Malaya can replace special words into tokens to reduce dimension
   curse. ``rm10k`` become ``<money>``.
2. Malaya can expand hashtags, ``#drmahathir`` become ``dr mahathir``.
3. Malaya can put tags for special words, ``#drmahathir`` become
   ``<hashtag> drmahathir </hashtag>``.
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
``RM10k``, the output is ``<money>``.

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
    string_2 = 'kecewa #johndoe movie and it suuuuucks!!! WASTED RM10... rm10 #badmovies :/'
    string_3 = "@husein:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    string_4 = 'aahhh, malasnye nak pegi keje harini #mondayblues'
    string_5 = '#drmahathir #najibrazak #1malaysia #mahathirnajib'

.. code:: ipython3

    %%time
    preprocessing = malaya.preprocessing.preprocessing()


.. parsed-literal::

    CPU times: user 16.6 s, sys: 3.49 s, total: 20.1 s
    Wall time: 24.9 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 9.22 ms, sys: 897 µs, total: 10.1 ms
    Wall time: 11.6 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 1.77 ms, sys: 35 µs, total: 1.81 ms
    Wall time: 1.81 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 809 µs, sys: 25 µs, total: 834 µs
    Wall time: 840 µs




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 1.89 ms, sys: 54 µs, total: 1.94 ms
    Wall time: 1.96 ms




.. parsed-literal::

    'aahh <elongated> , malas nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 7.36 ms, sys: 1.94 ms, total: 9.3 ms
    Wall time: 11.6 ms




.. parsed-literal::

    '<hashtag> dr mahathir </hashtag> <hashtag> najib razak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathir najib </hashtag>'



Load default paramaters with spelling correction to normalize elongated words.
------------------------------------------------------------------------------

We saw ``taak``, ``saabaar`` and another elongated words are not the
original words, so we can use spelling correction to normalize it.

.. code:: ipython3

    corrector = malaya.spell.probability()

.. code:: ipython3

    %%time
    preprocessing = malaya.preprocessing.preprocessing(speller = corrector)


.. parsed-literal::

    CPU times: user 16.6 s, sys: 3.04 s, total: 19.6 s
    Wall time: 23.2 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 192 ms, sys: 31.6 ms, total: 224 ms
    Wall time: 285 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> tak <elongated> sabar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 2.3 ms, sys: 695 µs, total: 2.99 ms
    Wall time: 2.8 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia sucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.06 ms, sys: 402 µs, total: 1.47 ms
    Wall time: 1.48 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 3.06 ms, sys: 402 µs, total: 3.46 ms
    Wall time: 4.03 ms




.. parsed-literal::

    'ah <elongated> , malas nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 5.11 ms, sys: 628 µs, total: 5.73 ms
    Wall time: 5.35 ms




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

    CPU times: user 177 ms, sys: 52.3 ms, total: 229 ms
    Wall time: 255 ms


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 2.69 ms, sys: 1.32 ms, total: 4.02 ms
    Wall time: 9.74 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathirmohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> davidlynch </hashtag> <hashtag> tvseries </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 1.41 ms, sys: 709 µs, total: 2.12 ms
    Wall time: 4.52 ms




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

    CPU times: user 15.5 s, sys: 3.23 s, total: 18.7 s
    Wall time: 22.1 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 80.2 ms, sys: 21.4 ms, total: 102 ms
    Wall time: 114 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 4.45 ms, sys: 2.89 ms, total: 7.34 ms
    Wall time: 10.6 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> movie and it suucks <elongated> ! <repeated> <allcaps> wasted </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.17 ms, sys: 102 µs, total: 1.27 ms
    Wall time: 1.92 ms




.. parsed-literal::

    '<user> : can not wait untuk the <date> <hashtag> sentiment </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



Tokenizer
---------

It able to tokenize multiple regex pipelines, you can check the list
from ``malaya.preprocessing.get_normalize()``

.. code:: ipython3

    tokenizer = malaya.preprocessing._SocialTokenizer().tokenize

.. code:: ipython3

    tokenizer(string_1)




.. parsed-literal::

    ['CANT',
     'WAIT',
     'for',
     'the',
     'new',
     'season',
     'of',
     '#mahathirmohamad',
     '＼(^o^)／',
     '!',
     '!',
     '!',
     '#davidlynch',
     '#tvseries',
     ':)))',
     ',',
     'TAAAK',
     'SAAABAAR',
     '!',
     '!',
     '!']



.. code:: ipython3

    tokenizer(string_2)




.. parsed-literal::

    ['kecewa',
     '#johndoe',
     'movie',
     'and',
     'it',
     'suuuuucks',
     '!',
     '!',
     '!',
     'WASTED',
     'RM10',
     '.',
     '.',
     '.',
     'rm10',
     '#badmovies',
     ':/']



.. code:: ipython3

    tokenizer(string_3)




.. parsed-literal::

    ['@husein',
     ':',
     'can',
     "'",
     't',
     'wait',
     'for',
     'the',
     'Nov 9',
     '#Sentiment',
     'talks',
     '!',
     'YAAAAAAY',
     '!',
     '!',
     '!',
     ':-D',
     'http://sentimentsymposium.com/.']



.. code:: ipython3

    tokenizer('saya nak makan ayam harga rm10k')




.. parsed-literal::

    ['saya', 'nak', 'makan', 'ayam', 'harga', 'rm10k']


