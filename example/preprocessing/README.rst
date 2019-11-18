
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 4.83 s, sys: 755 ms, total: 5.58 s
    Wall time: 5.38 s


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

    CPU times: user 15.5 s, sys: 3.12 s, total: 18.6 s
    Wall time: 21.6 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 78.1 ms, sys: 1.63 ms, total: 79.7 ms
    Wall time: 81.6 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 2.03 ms, sys: 43 µs, total: 2.07 ms
    Wall time: 2.11 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia suucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.58 ms, sys: 24 µs, total: 1.61 ms
    Wall time: 1.72 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 4.12 ms, sys: 759 µs, total: 4.88 ms
    Wall time: 4.26 ms




.. parsed-literal::

    'aahh <elongated> , malas nak pergi kerja hari ini <hashtag> isnin blues </hashtag>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 7.58 ms, sys: 855 µs, total: 8.43 ms
    Wall time: 7.82 ms




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

    CPU times: user 14.4 s, sys: 2.17 s, total: 16.6 s
    Wall time: 17.5 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 71.5 ms, sys: 14.9 ms, total: 86.4 ms
    Wall time: 88.1 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> tak <elongated> sabar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 2.01 ms, sys: 451 µs, total: 2.46 ms
    Wall time: 2.47 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> filem dan ia sucks <elongated> ! <repeated> <allcaps> dibazir </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 969 µs, sys: 319 µs, total: 1.29 ms
    Wall time: 1.3 ms




.. parsed-literal::

    '<user> : boleh tidak tunggu untuk yang <date> <hashtag> sentimen </hashtag> talks ! <allcaps> yay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 4.05 ms, sys: 731 µs, total: 4.78 ms
    Wall time: 4.77 ms




.. parsed-literal::

    'ah <elongated> , malas nak pergi kerja hari ini <hashtag> isnin blues </hashtag>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 9.05 ms, sys: 619 µs, total: 9.67 ms
    Wall time: 9.72 ms




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

    CPU times: user 163 ms, sys: 36.4 ms, total: 199 ms
    Wall time: 206 ms


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 1.25 ms, sys: 28 µs, total: 1.28 ms
    Wall time: 1.29 ms




.. parsed-literal::

    '<allcaps> tak boleh tunggu </allcaps> untuk yang baru musim daripada <hashtag> mahathirmohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> davidlynch </hashtag> <hashtag> tvseries </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 595 µs, sys: 24 µs, total: 619 µs
    Wall time: 628 µs




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

    CPU times: user 15 s, sys: 2.64 s, total: 17.6 s
    Wall time: 19.5 s


.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 85.9 ms, sys: 2.74 ms, total: 88.7 ms
    Wall time: 92 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 4.07 ms, sys: 942 µs, total: 5.01 ms
    Wall time: 4.66 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> movie and it suucks <elongated> ! <repeated> <allcaps> wasted </allcaps> <money> . <repeated> <money> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: ipython3

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.38 ms, sys: 74 µs, total: 1.46 ms
    Wall time: 1.48 ms




.. parsed-literal::

    '<user> : can not wait untuk the <date> <hashtag> sentiment </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



Tokenizer
---------

It able to tokenize multiple regex pipelines, you can check the list
from ``malaya.preprocessing.get_normalize()``

.. code:: ipython3

    tokenizer = malaya.preprocessing.SocialTokenizer().tokenize

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



Segmentation
------------

sometime when we want to clean social media texts or crawled texts, it
lack of spaces, example, ``sayanakmakannasiayam``,
``DrMahathir berjalan``.

We provide segmentation to split those sentences using Viterbi
algorithm.

.. code:: ipython3

    segmenter = malaya.preprocessing.segmenter()

.. code:: ipython3

    segmenter.segment('sayanakmakannasiayam')




.. parsed-literal::

    'saya nak makan nasi ayam'



.. code:: ipython3

    segmenter.segment('berjalandi')




.. parsed-literal::

    'berjalan di'



.. code:: ipython3

    segmenter.segment('DrMahathir')




.. parsed-literal::

    'dr mahathir'



But it will lower the output, you can create a simple function to fix
it.

.. code:: ipython3

    import re
    
    def segment(string):
        segmented = segmenter.segment(string)
        splitted = re.sub(r'[ ]+', ' ', segmented).strip().split()
        splitted = [split.title() if string[string.lower().find(split)].isupper() else split for split in splitted]
        return ' '.join(splitted)

.. code:: ipython3

    segment('DrMahathir dan NajibRazak')




.. parsed-literal::

    'Dr Mahathir dan Najib Razak'


