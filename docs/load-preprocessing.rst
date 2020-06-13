.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.22 s, sys: 1.35 s, total: 6.57 s
    Wall time: 8 s


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

.. code:: python

    string_1 = 'CANT WAIT for the new season of #mahathirmohamad ＼(^o^)／!!! #davidlynch #tvseries :))), TAAAK SAAABAAR!!!'
    string_2 = 'kecewa #johndoe movie and it suuuuucks!!! WASTED RM10... rm10 #badmovies :/'
    string_3 = "@husein:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    string_4 = 'aahhh, malasnye nak pegi keje harini #mondayblues'
    string_5 = '#drmahathir #najibrazak #1malaysia #mahathirnajib'

.. code:: python

    %%time
    preprocessing = malaya.preprocessing.preprocessing()


.. parsed-literal::

    CPU times: user 12.5 s, sys: 854 ms, total: 13.3 s
    Wall time: 13.4 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 6.71 ms, sys: 148 µs, total: 6.86 ms
    Wall time: 6.85 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 49.6 ms, sys: 847 µs, total: 50.4 ms
    Wall time: 51 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 546 µs, sys: 5 µs, total: 551 µs
    Wall time: 554 µs




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 1.83 ms, sys: 22 µs, total: 1.86 ms
    Wall time: 1.87 ms




.. parsed-literal::

    'aahh <elongated> , malas nak pergi kerja hari ini <hashtag> isnin blues </hashtag>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 4.91 ms, sys: 164 µs, total: 5.07 ms
    Wall time: 5.19 ms




.. parsed-literal::

    '<hashtag> dr mahathir </hashtag> <hashtag> najib razak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathir najib </hashtag>'



Load default paramaters with spelling correction to normalize elongated words.
------------------------------------------------------------------------------

We saw ``taak``, ``saabaar`` and another elongated words are not the
original words, so we can use spelling correction to normalize it.

.. code:: python

    corrector = malaya.spell.probability()

.. code:: python

    %%time
    preprocessing = malaya.preprocessing.preprocessing(speller = corrector)


.. parsed-literal::

    CPU times: user 12.4 s, sys: 888 ms, total: 13.3 s
    Wall time: 13.4 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 56.2 ms, sys: 1.29 ms, total: 57.5 ms
    Wall time: 58 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> tak <elongated> sabar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 1.24 ms, sys: 22 µs, total: 1.26 ms
    Wall time: 1.27 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia sucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 556 µs, sys: 8 µs, total: 564 µs
    Wall time: 567 µs




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 1.16 ms, sys: 14 µs, total: 1.17 ms
    Wall time: 1.19 ms




.. parsed-literal::

    'ah <elongated> , malas nak pergi kerja hari ini <hashtag> isnin blues </hashtag>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 3.82 ms, sys: 69 µs, total: 3.89 ms
    Wall time: 3.93 ms




.. parsed-literal::

    '<hashtag> dr mahathir </hashtag> <hashtag> najib razak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathir najib </hashtag>'



disable hashtag expander
------------------------

Sometime we want to maintain hashtags because we want to study the
frequencies.

.. code:: python

    %%time
    preprocessing = malaya.preprocessing.preprocessing(expand_hashtags = False)


.. parsed-literal::

    CPU times: user 93.7 ms, sys: 24 ms, total: 118 ms
    Wall time: 118 ms


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 907 µs, sys: 1e+03 ns, total: 908 µs
    Wall time: 913 µs




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathirmohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> davidlynch </hashtag> <hashtag> tvseries </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 616 µs, sys: 36 µs, total: 652 µs
    Wall time: 666 µs




.. parsed-literal::

    '<hashtag> drmahathir </hashtag> <hashtag> najibrazak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathirnajib </hashtag>'



disable english translation
---------------------------

But there are basic normalizations that cannot override, like, ``for``
automatically become ``untuk``. You can check default entire
normalizations at
``from malaya.texts._tatabahasa import rules_normalizer``

.. code:: python

    %%time
    preprocessing = malaya.preprocessing.preprocessing(translate_english_to_bm = False)


.. parsed-literal::

    CPU times: user 12.3 s, sys: 879 ms, total: 13.2 s
    Wall time: 13.2 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 53.4 ms, sys: 1.17 ms, total: 54.6 ms
    Wall time: 55 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 1.35 ms, sys: 18 µs, total: 1.37 ms
    Wall time: 1.39 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> movie and it suucks <elongated> ! <repeated> <allcaps> wasted </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 498 µs, sys: 1e+03 ns, total: 499 µs
    Wall time: 503 µs




.. parsed-literal::

    '<user> : can not wait untuk the <date> <hashtag> sentiment </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



Tokenizer
---------

It able to tokenize multiple regex pipelines, you can check the list
from ``malaya.preprocessing.get_normalize()``

.. code:: python

    tokenizer = malaya.preprocessing.SocialTokenizer().tokenize

.. code:: python

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



.. code:: python

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



.. code:: python

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



.. code:: python

    tokenizer('saya nak makan ayam harga rm10k')




.. parsed-literal::

    ['saya', 'nak', 'makan', 'ayam', 'harga', 'rm10k']



Segmentation
------------

sometime when we want to clean social media texts or crawled texts, it
lack of spaces, example, ``sayanakmakannasiayam``,
``DrMahathir berjalan``.

We provide segmentation to split those sentences using Viterbi
algorithm.

.. code:: python

    segmenter = malaya.preprocessing.segmenter()

.. code:: python

    segmenter.segment('sayanakmakannasiayam')




.. parsed-literal::

    'saya nak makan nasi ayam'



.. code:: python

    segmenter.segment('berjalandi')




.. parsed-literal::

    'berjalan di'



.. code:: python

    segmenter.segment('DrMahathir')




.. parsed-literal::

    'Dr Mahathir'



.. code:: python

    segmenter.segment('DRMahathir')




.. parsed-literal::

    'DR Mahathir'



.. code:: python

    segmenter.segment('drmahathirdannajibrazak')




.. parsed-literal::

    'dr mahathir dan najib razak'


