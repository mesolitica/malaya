

```python
import malaya
```

    deleting previous version models..


    1.00MB [00:00, 1.13kMB/s]                  

    downloading stopwords



    Using TensorFlow backend.
    /usr/local/lib/python3.5/dist-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    1.00MB [00:00, 15.0MB/s]                   

    downloading ZIP rules-based






```python
malaya.describe_pos()
```

    ADJ - Adjective, kata sifat
    ADP - Adposition
    ADV - Adverb, kata keterangan
    ADX - Auxiliary verb, kata kerja tambahan
    CCONJ - Coordinating conjuction, kata hubung
    DET - Determiner, kata penentu
    NOUN - Noun, kata nama
    NUM - Number, nombor
    PART - Particle
    PRON - Pronoun, kata ganti
    PROPN - Proper noun, kata ganti nama khas
    SCONJ - Subordinating conjunction
    SYM - Symbol
    VERB - Verb, kata kerja
    X - Other



```python
string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'
```


```python
malaya.multinomial_pos(string)
```

      0%|          | 0.00/0.90 [00:00<?, ?MB/s]

    downloading pickled bag-of-word multinomial POS


    1.00MB [00:00, 3.78MB/s]                   
      0%|          | 0.00/2.26 [00:00<?, ?MB/s]

    downloading pickled multinomial POS model


    3.00MB [00:00, 5.08MB/s]                          





    [('KUALA', 'ADV'),
     ('LUMPUR', 'ADV'),
     ('Sempena', 'ADJ'),
     ('sambutan', 'ADJ'),
     ('Aidilfitri', 'CCONJ'),
     ('minggu', 'ADJ'),
     ('depan', 'ADJ'),
     ('Perdana', 'CCONJ'),
     ('Menteri', 'ADJ'),
     ('Tun', 'CCONJ'),
     ('Dr', 'CCONJ'),
     ('Mahathir', 'CCONJ'),
     ('Mohamad', 'CCONJ'),
     ('dan', 'ADJ'),
     ('Menteri', 'ADJ'),
     ('Pengangkutan', 'ADJ'),
     ('Anthony', 'CCONJ'),
     ('Loke', 'CCONJ'),
     ('Siew', 'CCONJ'),
     ('Fook', 'AUX'),
     ('menitipkan', 'ADJ'),
     ('pesanan', 'ADJ'),
     ('khas', 'ADJ'),
     ('kepada', 'ADJ'),
     ('orang', 'ADJ'),
     ('ramai', 'ADJ'),
     ('yang', 'ADJ'),
     ('mahu', 'ADJ'),
     ('pulang', 'ADJ'),
     ('ke', 'ADJ'),
     ('kampung', 'ADJ'),
     ('halaman', 'ADJ'),
     ('masing-masing', 'ADJ'),
     ('Dalam', 'ADJ'),
     ('video', 'ADJ'),
     ('pendek', 'ADJ'),
     ('terbitan', 'ADJ'),
     ('Jabatan', 'AUX'),
     ('Keselamatan', 'ADJ'),
     ('Jalan', 'ADV'),
     ('Raya', 'CCONJ'),
     ('JKJR', 'ADJ'),
     ('itu', 'ADJ'),
     ('Dr', 'CCONJ'),
     ('Mahathir', 'CCONJ'),
     ('menasihati', 'ADJ'),
     ('mereka', 'ADJ'),
     ('supaya', 'ADJ'),
     ('berhenti', 'ADJ'),
     ('berehat', 'ADJ'),
     ('dan', 'ADJ'),
     ('tidur', 'ADJ'),
     ('sebentar', 'ADJ'),
     ('sekiranya', 'ADJ'),
     ('mengantuk', 'ADJ'),
     ('ketika', 'ADJ'),
     ('memandu', 'ADJ')]




```python
malaya.xgb_pos(string)
```

      0%|          | 0.00/1.21 [00:00<?, ?MB/s]

    downloading pickled bag-of-word XGB POS


    2.00MB [00:00, 6.26MB/s]                          
      0%|          | 0.00/86.3 [00:00<?, ?MB/s]

    downloading pickled xgb POS model


    87.0MB [00:23, 3.73MB/s]                          





    [('KUALA', 'PROPN'),
     ('LUMPUR', 'NOUN'),
     ('Sempena', 'PROPN'),
     ('sambutan', 'NOUN'),
     ('Aidilfitri', 'PROPN'),
     ('minggu', 'PROPN'),
     ('depan', 'ADJ'),
     ('Perdana', 'ADJ'),
     ('Menteri', 'PROPN'),
     ('Tun', 'PROPN'),
     ('Dr', 'PROPN'),
     ('Mahathir', 'PROPN'),
     ('Mohamad', 'PROPN'),
     ('dan', 'CCONJ'),
     ('Menteri', 'PROPN'),
     ('Pengangkutan', 'NOUN'),
     ('Anthony', 'PROPN'),
     ('Loke', 'NOUN'),
     ('Siew', 'PROPN'),
     ('Fook', 'PROPN'),
     ('menitipkan', 'VERB'),
     ('pesanan', 'NOUN'),
     ('khas', 'ADJ'),
     ('kepada', 'ADP'),
     ('orang', 'NOUN'),
     ('ramai', 'NOUN'),
     ('yang', 'PRON'),
     ('mahu', 'PROPN'),
     ('pulang', 'VERB'),
     ('ke', 'ADP'),
     ('kampung', 'NOUN'),
     ('halaman', 'NOUN'),
     ('masing-masing', 'NOUN'),
     ('Dalam', 'ADP'),
     ('video', 'NOUN'),
     ('pendek', 'ADJ'),
     ('terbitan', 'NOUN'),
     ('Jabatan', 'NOUN'),
     ('Keselamatan', 'NOUN'),
     ('Jalan', 'NOUN'),
     ('Raya', 'PROPN'),
     ('JKJR', 'NUM'),
     ('itu', 'DET'),
     ('Dr', 'PROPN'),
     ('Mahathir', 'PROPN'),
     ('menasihati', 'VERB'),
     ('mereka', 'PRON'),
     ('supaya', 'NOUN'),
     ('berhenti', 'VERB'),
     ('berehat', 'VERB'),
     ('dan', 'CCONJ'),
     ('tidur', 'NOUN'),
     ('sebentar', 'NOUN'),
     ('sekiranya', 'NOUN'),
     ('mengantuk', 'VERB'),
     ('ketika', 'SCONJ'),
     ('memandu', 'VERB')]




```python
for i in malaya.get_available_pos_models():
    print('Testing %s model'%(i))
    model = malaya.deep_pos(i)
    print(model.predict(string))
    print()
```

    1.00MB [00:00, 1.08kMB/s]                  

    Testing char model
    downloading POS char settings
    downloading POS frozen char model



    2.00MB [00:00, 6.92MB/s]                          


    [('kuala', 'PROPN'), ('lumpur', 'NOUN'), ('sempena', 'NOUN'), ('sambutan', 'NOUN'), ('aidilfitri', 'NOUN'), ('minggu', 'PROPN'), ('depan', 'ADJ'), ('perdana', 'PROPN'), ('menteri', 'NOUN'), ('tun', 'PROPN'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('mohamad', 'PROPN'), ('dan', 'CCONJ'), ('menteri', 'NOUN'), ('pengangkutan', 'NOUN'), ('anthony', 'PROPN'), ('loke', 'PROPN'), ('siew', 'PROPN'), ('fook', 'NOUN'), ('menitipkan', 'VERB'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'PROPN'), ('yang', 'PRON'), ('mahu', 'ADJ'), ('pulang', 'PROPN'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'NOUN'), ('dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'PROPN'), ('terbitan', 'NOUN'), ('jabatan', 'NOUN'), ('keselamatan', 'NOUN'), ('jalan', 'NOUN'), ('raya', 'PROPN'), ('jkjr', 'NOUN'), ('itu', 'DET'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('menasihati', 'VERB'), ('mereka', 'PRON'), ('supaya', 'NOUN'), ('berhenti', 'VERB'), ('berehat', 'VERB'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'SCONJ'), ('sekiranya', 'NOUN'), ('mengantuk', 'VERB'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing word model
    downloading POS word settings


    1.00MB [00:00, 4.62MB/s]                   
      0%|          | 0.00/9.94 [00:00<?, ?MB/s]

    downloading POS frozen word model


    10.0MB [00:02, 3.97MB/s]                          
      0%|          | 0.00/0.69 [00:00<?, ?MB/s]

    [('kuala', 'PROPN'), ('lumpur', 'PROPN'), ('sempena', 'PROPN'), ('sambutan', 'NOUN'), ('aidilfitri', 'PROPN'), ('minggu', 'PROPN'), ('depan', 'CCONJ'), ('perdana', 'PROPN'), ('menteri', 'NOUN'), ('tun', 'PROPN'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('mohamad', 'PROPN'), ('dan', 'CCONJ'), ('menteri', 'PROPN'), ('pengangkutan', 'PROPN'), ('anthony', 'PROPN'), ('loke', 'PROPN'), ('siew', 'PROPN'), ('fook', 'PROPN'), ('menitipkan', 'PROPN'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'PROPN'), ('dalam', 'ADP'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('jabatan', 'NOUN'), ('keselamatan', 'NOUN'), ('jalan', 'PROPN'), ('raya', 'PROPN'), ('jkjr', 'PROPN'), ('itu', 'DET'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('menasihati', 'PROPN'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'PROPN'), ('dan', 'CCONJ'), ('tidur', 'VERB'), ('sebentar', 'ADJ'), ('sekiranya', 'NOUN'), ('mengantuk', 'NOUN'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]

    Testing concat model
    downloading POS concat settings


    1.00MB [00:00, 4.66MB/s]                   
      0%|          | 0.00/5.44 [00:00<?, ?MB/s]

    downloading POS frozen concat model


    6.00MB [00:01, 4.34MB/s]                          


    [('kuala', 'PROPN'), ('lumpur', 'PROPN'), ('sempena', 'PROPN'), ('sambutan', 'NOUN'), ('aidilfitri', 'PROPN'), ('minggu', 'PROPN'), ('depan', 'ADJ'), ('perdana', 'PROPN'), ('menteri', 'NOUN'), ('tun', 'PROPN'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('mohamad', 'PROPN'), ('dan', 'CCONJ'), ('menteri', 'NOUN'), ('pengangkutan', 'PROPN'), ('anthony', 'PROPN'), ('loke', 'PROPN'), ('siew', 'PROPN'), ('fook', 'PROPN'), ('menitipkan', 'PROPN'), ('pesanan', 'NOUN'), ('khas', 'ADJ'), ('kepada', 'ADP'), ('orang', 'NOUN'), ('ramai', 'ADJ'), ('yang', 'PRON'), ('mahu', 'ADV'), ('pulang', 'VERB'), ('ke', 'ADP'), ('kampung', 'NOUN'), ('halaman', 'NOUN'), ('masing-masing', 'ADJ'), ('dalam', 'ADV'), ('video', 'NOUN'), ('pendek', 'ADJ'), ('terbitan', 'NOUN'), ('jabatan', 'NOUN'), ('keselamatan', 'PROPN'), ('jalan', 'PROPN'), ('raya', 'PROPN'), ('jkjr', 'PROPN'), ('itu', 'DET'), ('dr', 'PROPN'), ('mahathir', 'PROPN'), ('menasihati', 'NOUN'), ('mereka', 'PRON'), ('supaya', 'SCONJ'), ('berhenti', 'VERB'), ('berehat', 'PROPN'), ('dan', 'CCONJ'), ('tidur', 'NOUN'), ('sebentar', 'ADV'), ('sekiranya', 'NOUN'), ('mengantuk', 'ADJ'), ('ketika', 'SCONJ'), ('memandu', 'VERB')]
