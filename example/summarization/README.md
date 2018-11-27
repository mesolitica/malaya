

```python
import malaya
```

    Using TensorFlow backend.



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




    {'cluster-top-words': ['mei',
      'perintah',
      'alih',
      'sultan muhammad',
      'paduka titah',
      'peran',
      'buka',
      'malaysia'],
     'summary': 'dilantik pertuan agong dis kerajaan diterajui barisan nasional mei kepimpinan negara diambil alih pakatan harapan memenangi pilihan raya. merasmikan istiadat pembukaan penggal parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. peralihan kerajaan menyaksikan sultan muhammad v mencatat sejarah tersendiri pertuan agong malaysia memerintah era kerajaan berbeza',
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
malaya.summarize_lsa(isu_kerajaan, important_words=10,maintain_original=True)
```




    {'cluster-top-words': ['pilih',
      'mei',
      'alih',
      'sultan muhammad',
      'paduka titah',
      'peran',
      'buka',
      'malaysia'],
     'summary': 'semasa dilantik sebagai yang di-pertuan agong ke-15 pada 13 dis 2016, kerajaan ketika itu diterajui oleh barisan nasional dan pada 10 mei lepas, kepimpinan negara diambil alih oleh pakatan harapan yang memenangi pilihan raya umum ke-14. peralihan kerajaan itu menyaksikan sultan muhammad v mencatat sejarah tersendiri dengan menjadi yang di-pertuan agong malaysia yang pertama memerintah dalam era dua kerajaan berbeza. pada jun lepas, sultan muhammad v memperkenankan supaya peruntukan gaji dan emolumen yang di-pertuan agong dikurangkan sebanyak 10 peratus sepanjang pemerintahan sehingga 2021 berikutan keprihatinan seri paduka terhadap tahap hutang dan keadaan ekonomi negara',
     'top-words': ['titah',
      'pilih',
      'alih',
      'buka',
      'malaysia',
      'mei',
      'muhammad',
      'paduka titah',
      'peran',
      'sultan muhammad']}




```python
malaya.summarize_nmf(isu_kerajaan,important_words=10)
```




    {'cluster-top-words': ['mei',
      'perintah',
      'alih',
      'sultan muhammad',
      'paduka titah',
      'peran',
      'buka',
      'malaysia'],
     'summary': 'dilantik pertuan agong dis kerajaan diterajui barisan nasional mei kepimpinan negara diambil alih pakatan harapan memenangi pilihan raya. merasmikan istiadat pembukaan penggal parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. peralihan kerajaan menyaksikan sultan muhammad v mencatat sejarah tersendiri pertuan agong malaysia memerintah era kerajaan berbeza',
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
malaya.summarize_lda(isu_kerajaan,important_words=10)
```




    {'cluster-top-words': ['mei',
      'perintah',
      'alih',
      'sultan muhammad',
      'paduka titah',
      'peran',
      'buka',
      'malaysia'],
     'summary': 'dilantik pertuan agong dis kerajaan diterajui barisan nasional mei kepimpinan negara diambil alih pakatan harapan memenangi pilihan raya. merasmikan istiadat pembukaan penggal parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. peralihan kerajaan menyaksikan sultan muhammad v mencatat sejarah tersendiri pertuan agong malaysia memerintah era kerajaan berbeza',
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
malaya.summarize_lda(isu_kerajaan,important_words=10,return_cluster=False)
```




    {'summary': 'dilantik pertuan agong dis kerajaan diterajui barisan nasional mei kepimpinan negara diambil alih pakatan harapan memenangi pilihan raya. merasmikan istiadat pembukaan penggal parlimen julai seri paduka bertitah mengalu alukan pendekatan kerajaan pakatan harapan menegakkan ketelusan terutamanya mendedahkan kedudukan kewangan negara sebenar mengkaji perbelanjaan kos projek mengurus kewangan berhemat menangani kos sara hidup. peralihan kerajaan menyaksikan sultan muhammad v mencatat sejarah tersendiri pertuan agong malaysia memerintah era kerajaan berbeza',
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
deep_summary = malaya.summarize_deep_learning()
```


```python
deep_summary.summarize(isu_kerajaan)
```




    'peralihan kerajaan itu menyaksikan sultan muhammad v mencatat sejarah tersendiri dengan menjadi yang di-pertuan agong malaysia yang pertama memerintah dalam era dua kerajaan berbeza. semasa dilantik sebagai yang di-pertuan agong ke-15 pada 13 dis 2016, kerajaan ketika itu diterajui oleh barisan nasional dan pada 10 mei lepas, kepimpinan negara diambil alih oleh pakatan harapan yang memenangi pilihan raya umum ke-14. seri paduka turut menitahkan supaya majlis rumah terbuka aidilfitri tahun ini tidak diadakan di istana negara dengan peruntukan majlis itu digunakan bagi membantu golongan yang kurang bernasib baik'
