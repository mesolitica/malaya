
.. code:: python

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 12.5 s, sys: 1.56 s, total: 14 s
    Wall time: 19.7 s


Explanations
------------

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
       expand_contractions = True,
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
``mahathir mohamad``, but initial load is quite slow.

.. code:: python

    %%time
    preprocessing = malaya.preprocessing.preprocessing()


.. parsed-literal::

    CPU times: user 15.9 s, sys: 2.77 s, total: 18.7 s
    Wall time: 21.5 s


.. code:: python

    string_1 = 'CANT WAIT for the new season of #mahathirmohamad ＼(^o^)／!!! #davidlynch #tvseries :))), TAAAK SAAABAAR!!!'
    string_2 = 'kecewa #johndoe movie and it suuuuucks!!! WASTED RM10... #badmovies :/'
    string_3 = "@husein:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    string_4 = 'aahhh, malasnye nak pegi keje harini #mondayblues'
    string_5 = '#drmahathir #najibrazak #1malaysia #mahathirnajib'

.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 11.6 ms, sys: 2.88 ms, total: 14.4 ms
    Wall time: 16.3 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathir mohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> david lynch </hashtag> <hashtag> tv series </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_2))


.. parsed-literal::

    CPU times: user 4.09 ms, sys: 559 µs, total: 4.65 ms
    Wall time: 4.73 ms




.. parsed-literal::

    'kecewa <hashtag> john doe </hashtag> movie and it suucks <elongated> ! <repeated> <allcaps> wasted </allcaps> <money> <number> . <repeated> <hashtag> bad movies </hashtag> <annoyed>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_3))


.. parsed-literal::

    CPU times: user 1.35 ms, sys: 48 µs, total: 1.4 ms
    Wall time: 1.48 ms




.. parsed-literal::

    '<user> : can not wait untuk the <date> <hashtag> sentiment </hashtag> talks ! <allcaps> yaay <elongated> </allcaps> ! <repeated> :-d <url>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_4))


.. parsed-literal::

    CPU times: user 3.98 ms, sys: 1.54 ms, total: 5.52 ms
    Wall time: 8.04 ms




.. parsed-literal::

    'aahh <elongated> , malasnye nak pergi kerja hari ini <hashtag> monday blues </hashtag>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 11.4 ms, sys: 2.06 ms, total: 13.5 ms
    Wall time: 18.8 ms




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

    CPU times: user 249 µs, sys: 225 µs, total: 474 µs
    Wall time: 482 µs


.. code:: python

    %%time
    ' '.join(preprocessing.process(string_1))


.. parsed-literal::

    CPU times: user 1.83 ms, sys: 45 µs, total: 1.88 ms
    Wall time: 1.92 ms




.. parsed-literal::

    '<allcaps> tak boleh wait </allcaps> untuk the new season of <hashtag> mahathirmohamad </hashtag> \\(^o^)/ ! <repeated> <hashtag> davidlynch </hashtag> <hashtag> tvseries </hashtag> <happy> , <allcaps> taak <elongated> saabaar <elongated> </allcaps> ! <repeated>'



.. code:: python

    %%time
    ' '.join(preprocessing.process(string_5))


.. parsed-literal::

    CPU times: user 579 µs, sys: 29 µs, total: 608 µs
    Wall time: 628 µs




.. parsed-literal::

    '<hashtag> drmahathir </hashtag> <hashtag> najibrazak </hashtag> <hashtag> 1 malaysia </hashtag> <hashtag> mahathirnajib </hashtag>'
