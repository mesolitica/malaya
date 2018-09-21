

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
lda = malaya.lda_topic_modelling(corpus,10)
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

    bermakna strategi set dasar kondusif destinasi malaysia mestilah bijak berterusan strategi kempen pemasaran merancang pendekatan pasaran berbeza
    kerja keras ahli jentera parti dilaksanakan henti pru membuktikankesungguhan kepimpinan menabur bakti masyarakat ketinggalan manfaat disediakan kerajaan
    amanah bermasalah pas hati terbuka mengajak parti islam pakatan harapan harapan menyelamatkan negara
    politik perbezaan pandangan langsungkah titik persamaan pas parti pembangkang menjatuhkan kerajaan gagal mentadbir negara
    berfungsi semak imbang pakatan pelbagai isu berkaitan pergerakan sosial suara akar umbi terpinggir



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
    rakyat        keputusan     parti         hutang        menteri       
    malaysia      ambil         ros           mdb           perdana       
    negara        benda         umno          diselesaikan  berlaku       
    kepimpinan    kena          kebenaran     kewangan      jemaah        
    ppsmi         bersatu       pemilihan     tempoh        seri          
    kepentingan   peringkat     perlembagaan  pendek        penjelasan    
    memudahkan    negeri        kelulusan     wujud         isu           
    serius        ph            pendaftaran   projek        razak         
    negeri        sarawak       melebihi      tutup         najib         
    mengatasi     sabah         bersatu       bergantung    kuok          





```python
nmf.get_sentences(5)
```

    rakyat
    terpulang rakyat pertimbangkan
    rakyat malaysia kepentingan negara
    percaya berkongsi maklumat berasas terutamanya maklumat berkaitan kepimpinan negara rakyat malaysia rakyat memilih kepimpinan negara berkaliber sesuai pandangan rakyat
    memudahkan rakyat



```python
nmf.get_topics(10)
```




    [(0,
      'rakyat malaysia negara kepimpinan ppsmi kepentingan memudahkan serius negeri mengatasi'),
     (1, 'keputusan ambil benda kena bersatu peringkat negeri ph sarawak sabah'),
     (2,
      'parti ros umno kebenaran pemilihan perlembagaan kelulusan pendaftaran melebihi bersatu'),
     (3,
      'hutang mdb diselesaikan kewangan tempoh pendek wujud projek tutup bergantung'),
     (4, 'menteri perdana berlaku jemaah seri penjelasan isu razak najib kuok'),
     (5,
      'raya pilihan memandangkan kononnya perlembagaan kerusi prestasi kuok artikel mendakwa'),
     (6,
      'kerajaan negara meningkatkan pengangkutan tindakan malaysia pengajaran kemajuan bidang pembelajaran'),
     (7, 'kapal jho low dirampas perniagaan doj indonesia anak tuntutan sivil'),
     (8,
      'undi mengundi harapan pakatan catatan cina mendakwa demokrasi bahagian kepentingan'),
     (9,
      'berjalan projek lancar gembira pencarian peribadi pendidikan asalnya mengalami perdana')]




```python
lsa = malaya.lsa_topic_modelling(corpus,10)
lsa.print_topics(5)
```

    topic 0       topic 1       topic 2       topic 3       topic 4       
    --------      --------      --------      --------      --------      
    rakyat        rakyat        hutang        hutang        menteri       
    malaysia      malaysia      mdb           rakyat        perdana       
    negara        kepimpinan    negara        mdb           berlaku       
    kerajaan      ppsmi         projek        projek        kerajaan      
    parti         memudahkan    kewangan      diselesaikan  rakyat        
    isu           serius        diselesaikan  tempoh        jemaah        
    tindakan      dasar         kerajaan      kewangan      seri          
    menteri       berita        malaysia      pendek        penjelasan    
    berkongsi     kepentingan   kapal         tutup         asli          
    kepimpinan    mengatasi     low           sumber        isu           





```python
lsa.get_sentences(5)
```

    rakyat malaysia kepentingan negara
    rakyat malaysia celik serius isu kepimpinan negara negeri
    rakyat malaysia penyelesaian konkrit kerajaan mengatasi
    percaya berkongsi maklumat berasas terutamanya maklumat berkaitan kepimpinan negara rakyat malaysia rakyat memilih kepimpinan negara berkaliber sesuai pandangan rakyat
    terpulang rakyat pertimbangkan



```python
lsa.get_topics(10)
```




    [(0,
      'rakyat malaysia negara kerajaan parti isu tindakan menteri berkongsi kepimpinan'),
     (1,
      'rakyat malaysia kepimpinan ppsmi memudahkan serius dasar berita kepentingan mengatasi'),
     (2,
      'hutang mdb negara projek kewangan diselesaikan kerajaan malaysia kapal low'),
     (3,
      'hutang rakyat mdb projek diselesaikan tempoh kewangan pendek tutup sumber'),
     (4,
      'menteri perdana berlaku kerajaan rakyat jemaah seri penjelasan asli isu'),
     (5,
      'raya pilihan memandangkan perlembagaan kononnya prestasi kerusi kuok negara mendakwa'),
     (6, 'kerajaan jho kapal low perniagaan doj dirampas dana sivil ahli'),
     (7, 'kapal low jho menteri negara perdana doj berlaku anak perniagaan'),
     (8,
      'undi pertumbuhan asli harapan pendapatan pakatan mengundi masyarakat cina catatan'),
     (9, 'undi harapan pas isu pakatan amanah parti mdb menteri tindakan')]
