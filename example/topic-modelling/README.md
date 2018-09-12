

```python
import pandas as pd
import malaya
```

    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
df = pd.read_csv('tests/02032018.csv',sep=';')
df = df.iloc[3:,1:]
df.columns = ['text','label']
corpus = df.text.tolist()
```


```python
lda = malaya.lda_topic_modelling(corpus,10)
lda.get_topics(10)
```

    /usr/local/lib/python3.6/site-packages/sklearn/decomposition/online_lda.py:294: DeprecationWarning: n_topics has been renamed to n_components in version 0.19 and will be removed in 0.21
      DeprecationWarning)





    [(0, 'parti keputusan kelulusan umno tindakan mengambil faktor ph dana awam'),
     (1, 'pesawat berbeza dasar strategi terima mencari kempen kasih kuok rm'),
     (2,
      'kerajaan meningkatkan awam isu bahasa penjelasan kemajuan menerima mewujudkan proses'),
     (3,
      'projek hutang negara mdb kewangan diselesaikan syarikat malaysia wang sewa'),
     (4,
      'asli malaysia masyarakat bahasa jakoa pembangunan ambil perdana ilmu inggeris'),
     (5, 'air kapal low jho bersih rm selesa harga dana nik'),
     (6,
      'malaysia rakyat undi antarabangsa dasar memalukan bukti bertanding kenyataan proses'),
     (7,
      'pertumbuhan takut ros pesara tentera pekan pendapatan mca membenarkan jppm'),
     (8,
      'mdb perniagaan rakyat negara pilihan raya malaysia kepimpinan wujud hutang'),
     (9,
      'menteri perdana malaysia tumpuan najib seri jemaah arab keselamatan saudi')]




```python
nmf = malaya.nmf_topic_modelling(corpus,10)
nmf.get_topics(10)
```




    [(0,
      'rakyat malaysia negara kepimpinan ppsmi memudahkan kepentingan serius negeri mengatasi'),
     (1, 'keputusan ambil benda kena bersatu peringkat negeri ph sarawak sabah'),
     (2,
      'parti ros umno kebenaran pemilihan perlembagaan kelulusan pendaftaran melebihi bersatu'),
     (3,
      'hutang mdb diselesaikan kewangan tempoh pendek wujud projek tutup bergantung'),
     (4, 'menteri perdana berlaku jemaah seri penjelasan isu razak najib kuok'),
     (5,
      'raya pilihan memandangkan kononnya perlembagaan kerusi prestasi kuok artikel mendakwa'),
     (6,
      'kerajaan negara meningkatkan tindakan pengangkutan malaysia pengajaran kemajuan bidang pendidikan'),
     (7, 'kapal jho low dirampas perniagaan doj indonesia anak tuntutan sivil'),
     (8,
      'undi mengundi harapan pakatan catatan cina mendakwa demokrasi bahagian kepentingan'),
     (9,
      'berjalan projek lancar gembira pencarian peribadi pendidikan asalnya mengalami perdana')]




```python
lsa = malaya.lsa_topic_modelling(corpus,10)
lsa.get_topics(10)
```




    [(0,
      'rakyat malaysia negara kerajaan parti isu tindakan kepimpinan berkongsi menteri'),
     (1,
      'rakyat malaysia kepimpinan ppsmi memudahkan serius berita mengatasi kepentingan negeri'),
     (2, 'hutang mdb negara projek kewangan diselesaikan kerajaan kapal low jho'),
     (3,
      'hutang rakyat mdb projek diselesaikan tempoh kewangan pendek tutup wujud'),
     (4,
      'menteri perdana kerajaan rakyat berlaku seri jemaah penjelasan isu asli'),
     (5,
      'pilihan raya memandangkan perlembagaan kononnya prestasi kerusi kuok menerima pertumbuhan'),
     (6, 'kerajaan jho kapal low perniagaan dana doj dirampas ahli pas'),
     (7,
      'kerajaan hutang masyarakat meningkatkan asli pas pendidikan pembelajaran pengajaran proses'),
     (8,
      'undi asli harapan pendapatan cina pertumbuhan catatan masyarakat pakatan mengundi'),
     (9,
      'projek berjalan lancar ros gembira kebenaran asli pendidikan sumber pencarian')]




```python

```
