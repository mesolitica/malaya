
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.4 s, sys: 2.01 s, total: 14.4 s
    Wall time: 20.4 s


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

    CPU times: user 16.3 s, sys: 3.81 s, total: 20.2 s
    Wall time: 24.3 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 15.7 ms, sys: 9.26 ms, total: 25 ms
    Wall time: 24.4 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 2.7 ms, sys: 1.62 ms, total: 4.31 ms
    Wall time: 4.11 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 2.38 ms, sys: 707 µs, total: 3.09 ms
    Wall time: 3.25 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 4.43 ms, sys: 1.54 ms, total: 5.98 ms
    Wall time: 5.76 ms




.. parsed-literal::

    'aahh <elongated> , malas nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 13.1 ms, sys: 4.36 ms, total: 17.4 ms
    Wall time: 18.4 ms




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

    CPU times: user 15.2 s, sys: 3.64 s, total: 18.8 s
    Wall time: 21.3 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 219 ms, sys: 505 ms, total: 724 ms
    Wall time: 864 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> tak <elongated> sabar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 3.58 ms, sys: 2.23 ms, total: 5.81 ms
    Wall time: 6.66 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia sucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.12 ms, sys: 329 µs, total: 1.45 ms
    Wall time: 1.46 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 3.74 ms, sys: 1.35 ms, total: 5.1 ms
    Wall time: 6.45 ms




.. parsed-literal::

    'ah <elongated> , malas nak pergi kerja hari ini <hashtag> Isnin blues </hashtag>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 7.66 ms, sys: 1.74 ms, total: 9.4 ms
    Wall time: 12.2 ms




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

    CPU times: user 180 ms, sys: 51.6 ms, total: 232 ms
    Wall time: 253 ms


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 1.26 ms, sys: 71 µs, total: 1.33 ms
    Wall time: 1.34 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathirmohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> davidlynch </hashtag> <hashtag> tvseries </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 1.13 ms, sys: 730 µs, total: 1.86 ms
    Wall time: 1.64 ms




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

    CPU times: user 16 s, sys: 3.51 s, total: 19.5 s
    Wall time: 22.7 s


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 91.7 ms, sys: 102 ms, total: 194 ms
    Wall time: 259 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 5.73 ms, sys: 4.85 ms, total: 10.6 ms
    Wall time: 18.4 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> movie and it suucks <elongated> ! <repeated> <allcaps> wasted </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 958 µs, sys: 126 µs, total: 1.08 ms
    Wall time: 1.18 ms




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
lack of spaces, example, ``sayanakmakan nasi``, ``DrMahathir berjalan``.

We provide segmentation to split those sentences using Viterbi
algorithm.

.. code:: python

    segmenter = malaya.preprocessing.segmenter()

.. code:: python

    segmenter.segment('sayanakmakan nasiayam')




.. parsed-literal::

    'saya nak makan   nasi ayam'



.. code:: python

    segmenter.segment('berjalandi')




.. parsed-literal::

    'berjalan di'



.. code:: python

    segmenter.segment('DrMahathir')




.. parsed-literal::

    'dr mahathir'



But it will lower the output, you can create a simple function to fix
it.

.. code:: python

    import re

    def segment(string):
        segmented = segmenter.segment(string)
        splitted = re.sub(r'[ ]+', ' ', segmented).strip().split()
        splitted = [split.title() if string[string.lower().find(split)].isupper() else split for split in splitted]
        return ' '.join(splitted)

.. code:: python

    segment('DrMahathir dan NajibRazak')




.. parsed-literal::

    'Dr Mahathir dan Najib Razak'
