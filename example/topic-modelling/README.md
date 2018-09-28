

```python
import pandas as pd
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
df = pd.read_csv('tests/02032018.csv',sep=';')
df = df.iloc[3:,1:]
df.columns = ['text','label']
corpus = df.text.tolist()
```


```python
lda = malaya.lda_topic_modelling(corpus,10,stemming=False)
lda.print_topics(5)
```

    /usr/local/lib/python3.6/site-packages/sklearn/decomposition/online_lda.py:294: DeprecationWarning: n_topics has been renamed to n_components in version 0.19 and will be removed in 0.21
      DeprecationWarning)


    topic 0       topic 1       topic 2       topic 3       topic 4       
    --------      --------      --------      --------      --------      
    kerajaan      bahasa        negara        projek        diterjemahkan
    negara        keputusan     bank          faktor        ilmu          
    parti         raya          teknikal      parti         wujud         
    kelulusan     pilihan       malaysia      meningkatkan  dasar         
    malaysia      inggeris      pertumbuhan   harga         bahasa        
    tindakan      ilmu          berkongsi     nilai         bukti         
    pas           pengurusan    ekonomi       kerajaan      serahkan      
    undi          tanah         pengalaman    syarikat      mengumpul     
    masyarakat    sewa          selatan       tumpuan       asalnya       
    mengambil     parti         dibenarkan    bahasa        proses        





```python
lda.get_sentences(5)
```




    ['bermakna strategi set dasar kondusif destinasi malaysia mestilah bijak berterusan strategi kempen pemasaran merancang pendekatan pasaran berbeza',
     'kerja keras ahli jentera parti dilaksanakan henti pru membuktikankesungguhan kepimpinan menabur bakti masyarakat ketinggalan manfaat disediakan kerajaan',
     'amanah bermasalah pas hati terbuka mengajak parti islam pakatan harapan harapan menyelamatkan negara',
     'politik perbezaan pandangan langsungkah titik persamaan pas parti pembangkang menjatuhkan kerajaan gagal mentadbir negara',
     'berfungsi semak imbang pakatan pelbagai isu berkaitan pergerakan sosial suara akar umbi terpinggir']




```python
lda.get_topics(10)
```




    [(0,
      'kerajaan negara parti kelulusan malaysia tindakan pas undi masyarakat mengambil'),
     (1,
      'bahasa keputusan raya pilihan inggeris ilmu pengurusan tanah sewa parti'),
     (2,
      'negara bank teknikal malaysia pertumbuhan berkongsi ekonomi pengalaman selatan dibenarkan'),
     (3,
      'projek faktor parti meningkatkan harga nilai kerajaan syarikat tumpuan bahasa'),
     (4,
      'diterjemahkan ilmu wujud dasar bahasa bukti serahkan mengumpul asalnya proses'),
     (5, 'mdb low jho kenyataan ambil masyarakat sebarang kapal doj perniagaan'),
     (6,
      'awam ahli syarikat parti menteri pesawat berlaku tatakelakuan lembaga politik'),
     (7,
      'rakyat malaysia asli perniagaan pendapatan sumber menerima negara ros penambahbaikan'),
     (8,
      'hutang menteri mdb malaysia diselesaikan kewangan asli perdana isu negara'),
     (9,
      'wang tinggal memakan disediakan pekan kampung hasil hutang najib membabitkan')]




```python
nmf = malaya.nmf_topic_modelling(corpus,10)
nmf.print_topics(5)
```

    topic 0       topic 1       topic 2       topic 3       topic 4       
    --------      --------      --------      --------      --------      
    rakyat        ros           menteri       hutang        ambil         
    malaysia      tangguh       perdana       selesai       putus         
    kena          parti         terima        mdb           tindak        
    pimpin        umno          isu           wang          langkah       
    mudah         pilih         seri          wujud         dar           
    negara        lembaga       jemaah        tutup         mahkamah      
    negeri        putus         nyata         tempoh        punca         
    dasar         daftar        kena          projek        bahagian      
    serius        kelulus       razak         pendek        kelulus       
    tingkat       tempoh        raja          gantung       sarawak       





```python
nmf.get_sentences(5)
```




    ['rakyat malaysia negara',
     'kena rakyat',
     'mudah rakyat',
     'rakyat malaysia selesai konkrit raja',
     'rakyat malaysia celik serius isu pimpin negara negeri']




```python
nmf.get_topics(10)
```




    [(0, 'rakyat malaysia kena pimpin mudah negara negeri dasar serius tingkat'),
     (1, 'ros tangguh parti umno pilih lembaga putus daftar kelulus tempoh'),
     (2, 'menteri perdana terima isu seri jemaah nyata kena razak raja'),
     (3, 'hutang selesai mdb wang wujud tutup tempoh projek pendek gantung'),
     (4, 'ambil putus tindak langkah dar mahkamah punca bahagian kelulus sarawak'),
     (5, 'bangun negara kongsi malaysia alam sedia selatan kawasan main mahir'),
     (6, 'undi pakat impak tuju rana wujud catat cina dakwa sumber'),
     (7, 'kapal jho low rampas doj niaga dakwa sivil tuntut milik'),
     (8, 'laku ajar wujud bimbang proses raja didik bukti maju bidang'),
     (9, 'raya pilih konon lembaga selesai dengar pandang buah selesa kuok')]




```python
lsa = malaya.lsa_topic_modelling(corpus,10)
lsa.print_topics(5)
```

    topic 0       topic 1       topic 2       topic 3       topic 4       
    --------      --------      --------      --------      --------      
    rakyat        parti         laku          hutang        ambil         
    malaysia      pilih         hutang        mdb           putus         
    negara        ros           mdb           wang          undi          
    raja          tangguh       selesai       selesai       rana          
    menteri       umno          menteri       negara        tindak        
    parti         putus         projek        bangun        parti         
    selesai       lembaga       wujud         wujud         pas           
    kena          raya          terima        tutup         langkah       
    bangun        ambil         wang          tempoh        ph            
    ambil         daftar        nyata         pilih         kena          





```python
lsa.get_sentences(5)
```




    ['rakyat malaysia negara',
     'rakyat malaysia selesai konkrit raja',
     'rakyat malaysia celik serius isu pimpin negara negeri',
     'perdana menteri isu kena nyata raja terima',
     'percaya kongsi maklumat asas utama maklumat kait pimpin negara rakyat malaysia rakyat pilih pimpin negara kaliber sesuai pandang rakyat']




```python
lsa.get_topics(10)
```




    [(0, 'rakyat malaysia negara raja menteri parti selesai kena bangun ambil'),
     (1, 'parti pilih ros tangguh umno putus lembaga raya ambil daftar'),
     (2, 'laku hutang mdb selesai menteri projek wujud terima wang nyata'),
     (3, 'hutang mdb wang selesai negara bangun wujud tutup tempoh pilih'),
     (4, 'ambil putus undi rana tindak parti pas langkah ph kena'),
     (5, 'rakyat selesai hutang kena mdb malaysia menteri undi timbang wujud'),
     (6, 'undi laku ajar ros rakyat parti dakwa cina catat wujud'),
     (7, 'low jho kapal nyata dakwa rakyat niaga doj dana tumbuh'),
     (8, 'laku ambil ajar pilih dakwa bidang didik tindak putus raya'),
     (9, 'menteri bangun negara perdana laku putus kongsi alam parti nyata')]




```python

```
