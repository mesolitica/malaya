True Case
=========

.. container:: alert alert-info

   This tutorial is available as an IPython notebook at
   `Malaya/example/true-case <https://github.com/huseinzol05/Malaya/tree/master/example/true-case>`__.

.. code:: ipython3

    %%time
    
    import malaya


.. parsed-literal::

    CPU times: user 5.09 s, sys: 791 ms, total: 5.88 s
    Wall time: 5.61 s


Explanation
~~~~~~~~~~~

Common third party NLP services like Google Speech to Text or PDF to
Text will returned unsensitive case and no punctuations or mistake
punctuations and cases. So True Case can help you.

1. jom makan di us makanan di sana sedap -> jom makan di US, makanan di
   sana sedap.
2. kuala lumpur menteri di jabatan perdana menteri datuk seri dr mujahid
   yusof rawa hari ini mengakhiri lawatan kerja lapan hari ke jordan
   turki dan bosnia herzegovina lawatan yang bertujuan mengeratkan lagi
   hubungan dua hala dengan ketiga tiga negara berkenaan -> KUALA LUMPUR
   - Menteri di Jabatan Perdana Menteri, Datuk Seri Dr Mujahid Yusof
   Rawa hari ini mengakhiri lawatan kerja lapan hari ke Jordan, Turki
   dan Bosnia Herzegovina, lawatan yang bertujuan mengeratkan lagi
   hubungan dua hala dengan ketiga-tiga negara berkenaan.

True case only,

1. Solve mistake / no punctuations.
2. Solve mistake / unsensitive case.
3. Not correcting any grammar.

List available Transformer model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    malaya.true_case.available_transformer()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Size (MB)</th>
          <th>Sequence Accuracy</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>small</th>
          <td>42.7</td>
          <td>0.347</td>
        </tr>
        <tr>
          <th>base</th>
          <td>234.0</td>
          <td>0.696</td>
        </tr>
      </tbody>
    </table>
    </div>



Load Transformer model
----------------------

.. code:: ipython3

    model = malaya.true_case.transformer()


.. parsed-literal::

    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:73: The name tf.gfile.GFile is deprecated. Please use tf.io.gfile.GFile instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:75: The name tf.GraphDef is deprecated. Please use tf.compat.v1.GraphDef instead.
    
    WARNING:tensorflow:From /Users/huseinzolkepli/Documents/Malaya/malaya/function/__init__.py:68: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.
    


.. code:: ipython3

    string1 = 'jom makan di us makanan di sana sedap'
    string2 = 'kuala lumpur menteri di jabatan perdana menteri datuk seri dr mujahid yusof rawa hari ini mengakhiri lawatan kerja lapan hari ke jordan turki dan bosnia herzegovina lawatan yang bertujuan mengeratkan lagi hubungan dua hala dengan ketiga tiga negara berkenaan'

Predict
^^^^^^^

.. code:: ipython3

    from pprint import pprint

.. code:: ipython3

    pprint(model.true_case([string1, string2], beam_search = False))


.. parsed-literal::

    ['Jom makan di US makanan di sana sedap.',
     'KUALA LUMPUR - Menteri di Jabatan Perdana Menteri, Datuk Seri Dr Mujahid '
     'Yusof Rawa hari ini mengakhiri lawatan kerja lapan hari ke Jordan, Turki dan '
     'Bosnia Herzegovina, lawatan yang bertujuan mengeratkan lagi hubungan dua '
     'hala dengan ketiga-tiga negara berkenaan.']


.. code:: ipython3

    import random
    
    def random_uppercase(string):
        string = [c.upper() if random.randint(0,1) else c for c in string]
        return ''.join(string)

.. code:: ipython3

    r = random_uppercase(string2)
    r




.. parsed-literal::

    'KUAlA LUMpuR MeNtErI di jAbATan pErdANa mENTerI DATUk sERI dr muJaHid YusOf RAWA hARI ini mEnGAKhiri lAwaTan KERJa LapAN Hari Ke JordAn tURki daN BosNIa hErzEGOviNA lAwaTaN yANG BErtuJuan meNGEraTKan laGI hubuNGAN dua hAla deNgAn KetiGa TIGA nEGara BERkenAaN'



.. code:: ipython3

    pprint(model.true_case([r], beam_search = False))


.. parsed-literal::

    ['KUALA LUMPUR: Menteri di Jabatan Perdana Menteri, Datuk Seri Dr Mujahid '
     'Yusof Rawa hari ini mengakhiri lawatan kerja lapan hari ke Jordan, Turki dan '
     'Bosnia Herzegovina, lawatan yang bertujuan mengeratkan lagi hubungan dua '
     'hala dengan ketiga-tiga negara berkenaan.']

