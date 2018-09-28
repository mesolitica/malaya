

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
isu_kerajaan = ['Institusi raja khususnya Yang di-Pertuan Agong adalah kedaulatan negara dengan kedudukan dan peranannya termaktub dalam Perlembagaan Persekutuan yang perlu disokong dan didukung oleh kerajaan serta rakyat.',
               'Pensyarah Kulliyah Undang-Undang Ahmad Ibrahim, Universiti Islam Antarabangsa Malaysia (UIAM) Prof Madya Dr Shamrahayu Ab Aziz berkata perubahan kerajaan, susulan kemenangan Pakatan Harapan pada Pilihan Raya Umum Ke-14 pada Mei lepas, tidak memberi kesan dari segi peranan, fungsi dan kedudukan Yang di-Pertuan Agong.',
               'Peralihan kerajaan itu menyaksikan Sultan Muhammad V mencatat sejarah tersendiri dengan menjadi Yang di-Pertuan Agong Malaysia yang pertama memerintah dalam era dua kerajaan berbeza.',
               'Semasa dilantik sebagai Yang di-Pertuan Agong Ke-15 pada 13 Dis 2016, kerajaan ketika itu diterajui oleh Barisan Nasional dan pada 10 Mei lepas, kepimpinan negara diambil alih oleh Pakatan Harapan yang memenangi Pilihan Raya Umum Ke-14.',
               'Ketika merasmikan Istiadat Pembukaan Penggal Pertama, Parlimen ke-14 pada 17 Julai lepas, Seri Paduka bertitah mengalu-alukan pendekatan kerajaan Pakatan Harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup.',
               'Pada Jun lepas, Sultan Muhammad V memperkenankan supaya peruntukan gaji dan emolumen Yang di-Pertuan Agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan Seri Paduka terhadap tahap hutang dan keadaan ekonomi negara.',
               'Seri Paduka turut menitahkan supaya Majlis Rumah Terbuka Aidilfitri tahun ini tidak diadakan di Istana Negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik.']
```


```python
malaya.summarize_lsa(isu_kerajaan,important_words=10)
```




    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. pada jun sultan muhammad memperkenankan peruntukan gaji emolumen yang pertuan agong dikurangkan peratus pemerintahan berikutan keprihatinan seri paduka tahap hutang ekonomi negara. seri paduka menitahkan majlis rumah terbuka aidilfitri diadakan istana negara peruntukan majlis membantu golongan bernasib',
     'top-words': ['titah',
      'perintah',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'sultan muhammad',
      'peran'],
     'cluster-top-words': ['alih',
      'perintah',
      'buka',
      'peran',
      'malaysia',
      'mei',
      'sultan muhammad',
      'paduka titah']}




```python
malaya.summarize_lsa(isu_kerajaan, important_words=10,maintain_original=True)
```




    {'summary': 'Ketika merasmikan Istiadat Pembukaan Penggal Pertama, Parlimen ke-14 pada 17 Julai lepas, Seri Paduka bertitah mengalu-alukan pendekatan kerajaan Pakatan Harapan dalam menegakkan ketelusan terutamanya dengan mendedahkan kedudukan kewangan negara yang sebenar serta mengkaji semula perbelanjaan, kos projek dan mengurus kewangan secara berhemat bagi menangani kos sara hidup. Pada Jun lepas, Sultan Muhammad V memperkenankan supaya peruntukan gaji dan emolumen Yang di-Pertuan Agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan Seri Paduka terhadap tahap hutang dan keadaan ekonomi negara. Seri Paduka turut menitahkan supaya Majlis Rumah Terbuka Aidilfitri tahun ini tidak diadakan di Istana Negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik',
     'top-words': ['titah',
      'pilih',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'peran',
      'sultan muhammad'],
     'cluster-top-words': ['alih',
      'buka',
      'peran',
      'malaysia',
      'mei',
      'sultan muhammad',
      'paduka titah',
      'pilih']}




```python
malaya.summarize_nmf(isu_kerajaan,important_words=10)
```




    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. pada jun sultan muhammad memperkenankan peruntukan gaji emolumen yang pertuan agong dikurangkan peratus pemerintahan berikutan keprihatinan seri paduka tahap hutang ekonomi negara. seri paduka menitahkan majlis rumah terbuka aidilfitri diadakan istana negara peruntukan majlis membantu golongan bernasib',
     'top-words': ['titah',
      'perintah',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'sultan muhammad',
      'peran'],
     'cluster-top-words': ['alih',
      'perintah',
      'buka',
      'peran',
      'malaysia',
      'mei',
      'sultan muhammad',
      'paduka titah']}




```python
malaya.summarize_lda(isu_kerajaan,important_words=10)
```

    /usr/local/lib/python3.6/site-packages/sklearn/decomposition/online_lda.py:536: DeprecationWarning: The default value for 'learning_method' will be changed from 'online' to 'batch' in the release 0.20. This warning was introduced in 0.18.
      DeprecationWarning)





    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. pada jun sultan muhammad memperkenankan peruntukan gaji emolumen yang pertuan agong dikurangkan peratus pemerintahan berikutan keprihatinan seri paduka tahap hutang ekonomi negara. seri paduka menitahkan majlis rumah terbuka aidilfitri diadakan istana negara peruntukan majlis membantu golongan bernasib',
     'top-words': ['titah',
      'perintah',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'sultan muhammad',
      'peran'],
     'cluster-top-words': ['alih',
      'perintah',
      'buka',
      'peran',
      'malaysia',
      'mei',
      'sultan muhammad',
      'paduka titah']}




```python
malaya.summarize_lda(isu_kerajaan,important_words=10,return_cluster=False)
```

    /usr/local/lib/python3.6/site-packages/sklearn/decomposition/online_lda.py:536: DeprecationWarning: The default value for 'learning_method' will be changed from 'online' to 'batch' in the release 0.20. This warning was introduced in 0.18.
      DeprecationWarning)





    {'summary': 'ketika merasmikan istiadat pembukaan penggal pertama parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. pada jun sultan muhammad memperkenankan peruntukan gaji emolumen yang pertuan agong dikurangkan peratus pemerintahan berikutan keprihatinan seri paduka tahap hutang ekonomi negara. seri paduka menitahkan majlis rumah terbuka aidilfitri diadakan istana negara peruntukan majlis membantu golongan bernasib',
     'top-words': ['titah',
      'perintah',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'sultan muhammad',
      'peran']}




```python

```
