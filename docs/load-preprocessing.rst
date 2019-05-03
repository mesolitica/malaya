
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 13.9 s, sys: 1.77 s, total: 15.7 s
    Wall time: 23 s


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

    CPU times: user 15.5 s, sys: 1.61 s, total: 17.1 s
    Wall time: 18.9 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 10.6 ms, sys: 1.07 ms, total: 11.7 ms
    Wall time: 12.2 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 3.67 ms, sys: 1.86 ms, total: 5.53 ms
    Wall time: 6.26 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.23 ms, sys: 90 µs, total: 1.32 ms
    Wall time: 1.26 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 1.67 ms, sys: 34 µs, total: 1.7 ms
    Wall time: 1.74 ms




.. parsed-literal::

    'aahh <elongated> , malas nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 8.7 ms, sys: 634 µs, total: 9.33 ms
    Wall time: 15.5 ms




.. parsed-literal::

    '<hashtag> dr mahathir </hashtag> <hashtag> najib razak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathir najib </hashtag>'



Load default paramaters with spelling correction to normalize elongated words.
------------------------------------------------------------------------------

We saw ``taak``, ``saabaar`` and another elongated words are not the
original words, so we can use spelling correction to normalize it.

.. code:: python

    malays = malaya.load_malay_dictionary()
    corrector = malaya.spell.naive(malays)


.. parsed-literal::

    downloading Malay texts


.. parsed-literal::

    1.00MB [00:00, 9.40MB/s]


.. code:: python

    %%time
    preprocessing = malaya.preprocessing.preprocessing(speller = corrector)


.. parsed-literal::

    CPU times: user 16.2 s, sys: 2.53 s, total: 18.7 s
    Wall time: 22 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 670 ms, sys: 15.3 ms, total: 686 ms
    Wall time: 876 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> tawak <elongated> sabar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 109 ms, sys: 3.92 ms, total: 113 ms
    Wall time: 139 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 169 ms, sys: 4.66 ms, total: 174 ms
    Wall time: 240 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> ya <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 658 ms, sys: 12.4 ms, total: 670 ms
    Wall time: 822 ms




.. parsed-literal::

    'a <elongated> , malas nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 11.3 ms, sys: 1.89 ms, total: 13.2 ms
    Wall time: 53.9 ms




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

    CPU times: user 205 ms, sys: 43.5 ms, total: 248 ms
    Wall time: 280 ms


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 1.34 ms, sys: 16 µs, total: 1.36 ms
    Wall time: 1.39 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathirmohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> davidlynch </hashtag> <hashtag> tvseries </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 1.17 ms, sys: 6 µs, total: 1.17 ms
    Wall time: 1.25 ms




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

    CPU times: user 18 s, sys: 3.49 s, total: 21.5 s
    Wall time: 30.4 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 74.2 ms, sys: 5.8 ms, total: 80 ms
    Wall time: 85.8 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 1.91 ms, sys: 148 µs, total: 2.06 ms
    Wall time: 2.06 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> movie and it suucks <elongated> ! <repeated> <allcaps> wasted </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.3 ms, sys: 129 µs, total: 1.43 ms
    Wall time: 1.61 ms




.. parsed-literal::

    '<user> : can not wait untuk the <date> <hashtag> sentiment </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



Tokenizer
---------

It able to tokenize multiple regex pipelines, you can check the list
from ``malaya.preprocessing.get_normalize()``

.. code:: python

    tokenizer = malaya.preprocessing._SocialTokenizer().tokenize

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
