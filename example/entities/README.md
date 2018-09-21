

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'
```


```python
malaya.describe_pos_malaya()
```

    KT - Kata Tanya
    KJ - Kata Kerja
    KP - Kata Perintah
    KPA - Kata Pangkal
    KB - Kata Bantu
    KPENGUAT - Kata Penguat
    KPENEGAS - Kata Penegas
    NAFI - Kata Nafi
    KPEMERI - Kata Pemeri
    KS - Kata Sendi
    KPEMBENAR - Kata Pembenar
    NAFI - Kata Nafi
    NO - Numbers
    SUKU - Suku Bilangan
    PISAHAN - Kata Pisahan
    KETERANGAN - Kata Keterangan
    ARAH - Kata Arah
    KH - Kata Hubung
    GN - Ganti Nama
    KA - Kata Adjektif
    O - not related, out scope



```python
malaya.describe_entities_malaya()
```

    PRN - person, group of people, believes, etc
    LOC - location
    NORP - Military, police, government, Parties, etc
    ORG - Organization, company
    LAW - related law document, etc
    ART - art of work, special names, etc
    EVENT - event happen, etc
    FAC - facility, hospitals, clinics, etc
    TIME - date, day, time, etc
    O - not related, out scope



```python
malaya.naive_pos(string)
```




    [('KN', 'kuala'),
     ('KN', 'lumpur'),
     ('', ':'),
     ('KJ', 'sempena'),
     ('KN', 'sambutan'),
     ('KJ', 'aidilfitri'),
     ('KN', 'minggu'),
     ('KN', 'depan'),
     ('', ','),
     ('KJ', 'perdana'),
     ('KJ', 'menteri'),
     ('KN', 'tun'),
     ('', 'dr'),
     ('KN', 'mahathir'),
     ('KN', 'mohamad'),
     ('KH', 'dan'),
     ('KJ', 'menteri'),
     ('KJ', 'pengangkutan'),
     ('KN', 'anthony'),
     ('KN', 'loke'),
     ('KN', 'siew'),
     ('KN', 'fook'),
     ('KJ', 'menitipkan'),
     ('KJ', 'pesanan'),
     ('KN', 'khas'),
     ('KS', 'kepada'),
     ('KN', 'orang'),
     ('KN', 'ramai'),
     ('KETERANGAN', 'yang'),
     ('KN', 'mahu'),
     ('KN', 'pulang'),
     ('KS', 'ke'),
     ('KN', 'kampung'),
     ('KN', 'halaman'),
     ('KN', 'masing-masing'),
     ('', '.'),
     ('KS', 'dalam'),
     ('KN', 'video'),
     ('KJ', 'pendek'),
     ('KJ', 'terbitan'),
     ('KN', 'jabatan'),
     ('KJ', 'keselamatan'),
     ('KN', 'jalan'),
     ('KN', 'raya'),
     ('', '('),
     ('KN', 'jkjr'),
     ('', ')'),
     ('GN', 'itu'),
     ('', ','),
     ('', 'dr'),
     ('KN', 'mahathir'),
     ('KJ', 'menasihati'),
     ('GN', 'mereka'),
     ('KH', 'supaya'),
     ('KJ', 'berhenti'),
     ('KJ', 'berehat'),
     ('KH', 'dan'),
     ('KN', 'tidur'),
     ('KETERANGAN', 'sebentar'),
     ('KJ', 'sekiranya'),
     ('KJ', 'mengantuk'),
     ('KN', 'ketika'),
     ('KJ', 'memandu'),
     ('', '.')]




```python
available_models = malaya.get_available_pos_entities_models()
available_models
```




    ['char', 'concat', 'attention']




```python
for i in available_models:
    print('Testing %s model'%(i))
    print(malaya.deep_pos_entities(i).predict(string))
    print()
```

    Testing char model
    [('KUALA', 'LOC', 'KN'), ('LUMPUR:', 'LOC', 'KN'), ('Sempena', 'O', 'KN'), ('sambutan', 'O', 'KN'), ('Aidilfitri', 'EVENT', 'KN'), ('minggu', 'O', 'KN'), ('depan,', 'O', 'KN'), ('Perdana', 'PRN', 'KN'), ('Menteri', 'PRN', 'KN'), ('Tun', 'PRN', 'KN'), ('Dr', 'PRN', 'KN'), ('Mahathir', 'PRN', 'KN'), ('Mohamad', 'PRN', 'KN'), ('dan', 'O', 'KH'), ('Menteri', 'PRN', 'KN'), ('Pengangkutan', 'LAW', 'KN'), ('Anthony', 'PRN', 'KN'), ('Loke', 'PRN', 'KN'), ('Siew', 'PRN', 'KN'), ('Fook', 'PRN', 'KN'), ('menitipkan', 'O', 'KN'), ('pesanan', 'O', 'KN'), ('khas', 'O', 'KN'), ('kepada', 'O', 'KS'), ('orang', 'O', 'KN'), ('ramai', 'O', 'KN'), ('yang', 'O', 'KETERANGAN'), ('mahu', 'O', 'KN'), ('pulang', 'O', 'KN'), ('ke', 'O', 'KS'), ('kampung', 'LOC', 'KN'), ('halaman', 'LOC', 'KN'), ('masing-masing.', 'O', 'KN'), ('Dalam', 'O', 'KS'), ('video', 'O', 'KN'), ('pendek', 'O', 'KN'), ('terbitan', 'O', 'KN'), ('Jabatan', 'NORP', 'KN'), ('Keselamatan', 'O', 'KN'), ('Jalan', 'LOC', 'KN'), ('Raya', 'ART', 'KN'), ('(JKJR)', 'PRN', 'KN'), ('itu,', 'O', 'GN'), ('Dr', 'PRN', 'KN'), ('Mahathir', 'PRN', 'KN'), ('menasihati', 'O', 'KJ'), ('mereka', 'O', 'GN'), ('supaya', 'O', 'KH'), ('berhenti', 'O', 'KJ'), ('berehat', 'O', 'KA'), ('dan', 'O', 'KH'), ('tidur', 'O', 'KN'), ('sebentar', 'O', 'KETERANGAN'), ('sekiranya', 'O', 'KN'), ('mengantuk', 'O', 'KJ'), ('ketika', 'O', 'KN'), ('memandu.', 'O', 'KJ')]

    Testing concat model
    [('KUALA', 'LOC', 'KN'), ('LUMPUR:', 'LOC', 'KN'), ('Sempena', 'O', 'KN'), ('sambutan', 'EVENT', 'KN'), ('Aidilfitri', 'EVENT', 'KN'), ('minggu', 'O', 'KN'), ('depan,', 'O', 'KN'), ('Perdana', 'PRN', 'KN'), ('Menteri', 'PRN', 'KN'), ('Tun', 'PRN', 'KN'), ('Dr', 'PRN', 'KN'), ('Mahathir', 'PRN', 'KN'), ('Mohamad', 'PRN', 'KN'), ('dan', 'O', 'KH'), ('Menteri', 'O', 'KN'), ('Pengangkutan', 'O', 'KN'), ('Anthony', 'PRN', 'KN'), ('Loke', 'PRN', 'KN'), ('Siew', 'PRN', 'KN'), ('Fook', 'PRN', 'KN'), ('menitipkan', 'O', 'KN'), ('pesanan', 'O', 'KN'), ('khas', 'O', 'KN'), ('kepada', 'O', 'KS'), ('orang', 'O', 'KN'), ('ramai', 'O', 'KN'), ('yang', 'O', 'KETERANGAN'), ('mahu', 'O', 'KN'), ('pulang', 'O', 'KN'), ('ke', 'O', 'KS'), ('kampung', 'O', 'KN'), ('halaman', 'TIME', 'KN'), ('masing-masing.', 'TIME', 'KN'), ('Dalam', 'O', 'KN'), ('video', 'O', 'KN'), ('pendek', 'O', 'KN'), ('terbitan', 'O', 'KN'), ('Jabatan', 'NORP', 'KN'), ('Keselamatan', 'NORP', 'KN'), ('Jalan', 'O', 'KN'), ('Raya', 'ART', 'KN'), ('(JKJR)', 'ART', 'KN'), ('itu,', 'O', 'GN'), ('Dr', 'PRN', 'KN'), ('Mahathir', 'PRN', 'KN'), ('menasihati', 'PRN', 'KN'), ('mereka', 'O', 'GN'), ('supaya', 'O', 'KH'), ('berhenti', 'O', 'KJ'), ('berehat', 'O', 'KN'), ('dan', 'O', 'KH'), ('tidur', 'O', 'KN'), ('sebentar', 'O', 'KA'), ('sekiranya', 'O', 'KJ'), ('mengantuk', 'O', 'KN'), ('ketika', 'O', 'KN'), ('memandu.', 'O', 'KJ')]

    Testing attention model
    [('KUALA', 'LOC', 'KN'), ('LUMPUR:', 'LOC', 'KN'), ('Sempena', 'O', 'KN'), ('sambutan', 'O', 'KN'), ('Aidilfitri', 'EVENT', 'KN'), ('minggu', 'O', 'KN'), ('depan,', 'O', 'KN'), ('Perdana', 'PRN', 'KN'), ('Menteri', 'PRN', 'KN'), ('Tun', 'PRN', 'KN'), ('Dr', 'PRN', 'KN'), ('Mahathir', 'PRN', 'KN'), ('Mohamad', 'PRN', 'KN'), ('dan', 'O', 'KH'), ('Menteri', 'NORP', 'KN'), ('Pengangkutan', 'NORP', 'KN'), ('Anthony', 'PRN', 'KN'), ('Loke', 'PRN', 'KN'), ('Siew', 'PRN', 'KN'), ('Fook', 'PRN', 'KN'), ('menitipkan', 'O', 'KN'), ('pesanan', 'O', 'KN'), ('khas', 'O', 'KN'), ('kepada', 'O', 'KS'), ('orang', 'O', 'KN'), ('ramai', 'O', 'KN'), ('yang', 'O', 'KETERANGAN'), ('mahu', 'O', 'KN'), ('pulang', 'O', 'KN'), ('ke', 'O', 'KS'), ('kampung', 'O', 'KN'), ('halaman', 'O', 'KN'), ('masing-masing.', 'O', 'KN'), ('Dalam', 'O', 'KN'), ('video', 'O', 'KN'), ('pendek', 'O', 'KN'), ('terbitan', 'O', 'KN'), ('Jabatan', 'O', 'KN'), ('Keselamatan', 'O', 'KN'), ('Jalan', 'NORP', 'KN'), ('Raya', 'NORP', 'KN'), ('(JKJR)', 'O', 'KN'), ('itu,', 'O', 'GN'), ('Dr', 'PRN', 'KN'), ('Mahathir', 'PRN', 'KN'), ('menasihati', 'O', 'KN'), ('mereka', 'O', 'GN'), ('supaya', 'O', 'KH'), ('berhenti', 'O', 'KJ'), ('berehat', 'O', 'KN'), ('dan', 'O', 'KH'), ('tidur', 'O', 'KN'), ('sebentar', 'O', 'KETERANGAN'), ('sekiranya', 'O', 'KJ'), ('mengantuk', 'O', 'KN'), ('ketika', 'O', 'KN'), ('memandu.', 'O', 'KJ')]




```python

```
