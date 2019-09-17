
.. code:: ipython3

    %%time
    import malaya


.. parsed-literal::

    CPU times: user 5.37 s, sys: 841 ms, total: 6.21 s
    Wall time: 6.97 s


.. code:: ipython3

    # some text examples copied from Twitter
    
    string1 = 'kerajaan patut bagi pencen awal skt kpd warga emas supaya emosi'
    string2 = 'Husein ska mkn aym dkat kmpng Jawa'
    string3 = 'Melayu malas ni narration dia sama je macam men are trash. True to some, false to some.'
    string4 = 'Tapi tak pikir ke bahaya perpetuate myths camtu. Nanti kalau ada hiring discrimination despite your good qualifications because of your race tau pulak marah. Your kids will be victims of that too.'
    string5 = 'DrM cerita Melayu malas semenjak saya kat University (early 1980s) and now as i am edging towards retirement in 4-5 years time after a career of being an Engineer, Project Manager, General Manager'
    string6 = 'blh bntg dlm kls nlp sy, nnti intch'

Load probability speller
------------------------

The probability speller extends the functionality of the Peter Norvig’s,
http://norvig.com/spell-correct.html.

And improve it using some algorithms from Normalization of noisy texts
in Malaysian online reviews,
https://www.researchgate.net/publication/287050449_Normalization_of_noisy_texts_in_Malaysian_online_reviews.

Also added custom vowels and consonant augmentation to adapt with our
local shortform / typos.

.. code:: ipython3

    prob_corrector = malaya.spell.probability()

To correct a word
^^^^^^^^^^^^^^^^^

.. code:: ipython3

    prob_corrector.correct('sy')




.. parsed-literal::

    'saya'



.. code:: ipython3

    prob_corrector.correct('mhthir')




.. parsed-literal::

    'mahathir'



.. code:: ipython3

    prob_corrector.correct('mknn')




.. parsed-literal::

    'makanan'



List possible generated pool of words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    prob_corrector.edit_candidates('mhthir')




.. parsed-literal::

    {'mahathir'}



.. code:: ipython3

    prob_corrector.edit_candidates('smbng')




.. parsed-literal::

    {'sambang',
     'sambong',
     'sambung',
     'sembang',
     'sembong',
     'sembung',
     'simbang',
     'smbg',
     'sombong',
     'sumbang',
     'sumbing'}



**So how does the model knows which words need to pick? highest counts
from wikipedia!**

To correct a sentence
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    prob_corrector.correct_text(string1)




.. parsed-literal::

    'kerajaan patut bagi pencen awal sakit kepada warga emas supaya emosi'



.. code:: ipython3

    prob_corrector.correct_text(string2)




.. parsed-literal::

    'Husein suka makan ayam dekat kmpng Jawa'



.. code:: ipython3

    prob_corrector.correct_text(string3)




.. parsed-literal::

    'Melayu malas ni narration dia sama sahaja macam men are trash. True to some, false to some.'



.. code:: ipython3

    prob_corrector.correct_text(string4)




.. parsed-literal::

    'Tapi tak fikir ke bahaya perpetuate myths macam itu. Nanti kalau ada hiring discrimination despite your good qualifications because of your race tahu pula marah. Your kids will be victims of that too.'



.. code:: ipython3

    prob_corrector.correct_text(string5)




.. parsed-literal::

    'DrM cerita Melayu malas semenjak saya kat University (early 1980s) and now as saya am edging towards retirement in 4-5 years time after a career of being an Engineer, Project Manager, General Manager'



.. code:: ipython3

    prob_corrector.correct_text(string6)




.. parsed-literal::

    'boleh bintang dalam kelas nlp saya, nanti intch'



Load symspeller speller
-----------------------

This spelling correction is an improvement version for
`symspeller <https://github.com/mammothb/symspellpy>`__ to adapt with
our local shortform / typos. Before you able to use this spelling
correction, you need to install
`symspeller <https://github.com/mammothb/symspellpy>`__,

.. code:: bash

   pip install symspellpy

.. code:: ipython3

    symspell_corrector = malaya.spell.symspell()

To correct a word
^^^^^^^^^^^^^^^^^

.. code:: ipython3

    symspell_corrector.correct('bntng')




.. parsed-literal::

    'bintang'



.. code:: ipython3

    symspell_corrector.correct('kerajaan')




.. parsed-literal::

    'kerajaan'



.. code:: ipython3

    symspell_corrector.correct('mknn')




.. parsed-literal::

    'makanan'



List possible generated words
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    symspell_corrector.edit_step('mrh')




.. parsed-literal::

    {'marah': 12684.0,
     'merah': 21448.5,
     'arah': 15066.5,
     'darah': 10003.0,
     'mara': 7504.5,
     'malah': 7450.0,
     'zarah': 3753.5,
     'murah': 3575.5,
     'barah': 2707.5,
     'march': 2540.5,
     'martha': 390.0,
     'marsha': 389.0,
     'maratha': 88.5,
     'marcha': 22.5,
     'karaha': 13.5,
     'maraba': 13.5,
     'varaha': 11.5,
     'marana': 4.5,
     'marama': 4.5}



To correct a sentence
^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    symspell_corrector.correct_text(string1)




.. parsed-literal::

    'kerajaan patut bagi pencen awal saat kepada warga emas supaya emosi'



.. code:: ipython3

    symspell_corrector.correct_text(string2)




.. parsed-literal::

    'Husein sama makan ayam dapat kompang Jawa'



.. code:: ipython3

    symspell_corrector.correct_text(string3)




.. parsed-literal::

    'Melayu malas ni narration dia sama sahaja macam men are trash. True to some, false to some.'



.. code:: ipython3

    symspell_corrector.correct_text(string4)




.. parsed-literal::

    'Tapi tak fikir ke bahaya perpetuate maathai macam itu. Nanti kalau ada hiring discrimination despite your good qualifications because of your race tahu pula marah. Your kids will be victims of that too.'



.. code:: ipython3

    symspell_corrector.correct_text(string5)




.. parsed-literal::

    'DrM cerita Melayu malas semenjak saya kat University (early 1980s) and now as saya am edging towards retirement in 4-5 aras time after a career of being an Engineer, Project Manager, General Manager'



.. code:: ipython3

    symspell_corrector.correct_text(string6)




.. parsed-literal::

    'ialah bintang dalam kelas malaya saya, nanti mintalah'


