

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
malaya.describe_entities()
```

    OTHER - not related, out of scope
    law - documents, law related
    location - location, place
    organization - Organization, company, government, parties
    person - person, group of people, believes, special names
    quantity - countable
    time - date, day, time



```python
string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'
```


```python
malaya.multinomial_entities(string)
```

    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator CountVectorizer from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator MultinomialNB from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)





    [('KUALA', 'location'),
     ('LUMPUR', 'location'),
     ('Sempena', 'OTHER'),
     ('sambutan', 'OTHER'),
     ('Aidilfitri', 'person'),
     ('minggu', 'OTHER'),
     ('depan', 'OTHER'),
     ('Perdana', 'person'),
     ('Menteri', 'OTHER'),
     ('Tun', 'person'),
     ('Dr', 'person'),
     ('Mahathir', 'person'),
     ('Mohamad', 'person'),
     ('dan', 'OTHER'),
     ('Menteri', 'OTHER'),
     ('Pengangkutan', 'OTHER'),
     ('Anthony', 'person'),
     ('Loke', 'person'),
     ('Siew', 'person'),
     ('Fook', 'organization'),
     ('menitipkan', 'OTHER'),
     ('pesanan', 'OTHER'),
     ('khas', 'OTHER'),
     ('kepada', 'OTHER'),
     ('orang', 'OTHER'),
     ('ramai', 'OTHER'),
     ('yang', 'OTHER'),
     ('mahu', 'OTHER'),
     ('pulang', 'OTHER'),
     ('ke', 'OTHER'),
     ('kampung', 'OTHER'),
     ('halaman', 'OTHER'),
     ('masing-masing', 'OTHER'),
     ('Dalam', 'OTHER'),
     ('video', 'OTHER'),
     ('pendek', 'OTHER'),
     ('terbitan', 'OTHER'),
     ('Jabatan', 'organization'),
     ('Keselamatan', 'OTHER'),
     ('Jalan', 'location'),
     ('Raya', 'person'),
     ('JKJR', 'OTHER'),
     ('itu', 'OTHER'),
     ('Dr', 'person'),
     ('Mahathir', 'person'),
     ('menasihati', 'OTHER'),
     ('mereka', 'OTHER'),
     ('supaya', 'OTHER'),
     ('berhenti', 'OTHER'),
     ('berehat', 'OTHER'),
     ('dan', 'OTHER'),
     ('tidur', 'OTHER'),
     ('sebentar', 'OTHER'),
     ('sekiranya', 'OTHER'),
     ('mengantuk', 'OTHER'),
     ('ketika', 'OTHER'),
     ('memandu', 'OTHER')]




```python
malaya.xgb_entities(string)
```

    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator CountVectorizer from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)





    [('KUALA', 'location'),
     ('LUMPUR', 'location'),
     ('Sempena', 'OTHER'),
     ('sambutan', 'OTHER'),
     ('Aidilfitri', 'OTHER'),
     ('minggu', 'time'),
     ('depan', 'OTHER'),
     ('Perdana', 'person'),
     ('Menteri', 'OTHER'),
     ('Tun', 'person'),
     ('Dr', 'person'),
     ('Mahathir', 'person'),
     ('Mohamad', 'person'),
     ('dan', 'OTHER'),
     ('Menteri', 'OTHER'),
     ('Pengangkutan', 'OTHER'),
     ('Anthony', 'person'),
     ('Loke', 'OTHER'),
     ('Siew', 'person'),
     ('Fook', 'OTHER'),
     ('menitipkan', 'OTHER'),
     ('pesanan', 'OTHER'),
     ('khas', 'organization'),
     ('kepada', 'OTHER'),
     ('orang', 'OTHER'),
     ('ramai', 'OTHER'),
     ('yang', 'OTHER'),
     ('mahu', 'OTHER'),
     ('pulang', 'OTHER'),
     ('ke', 'OTHER'),
     ('kampung', 'OTHER'),
     ('halaman', 'OTHER'),
     ('masing-masing', 'OTHER'),
     ('Dalam', 'OTHER'),
     ('video', 'OTHER'),
     ('pendek', 'OTHER'),
     ('terbitan', 'OTHER'),
     ('Jabatan', 'OTHER'),
     ('Keselamatan', 'OTHER'),
     ('Jalan', 'location'),
     ('Raya', 'OTHER'),
     ('JKJR', 'OTHER'),
     ('itu', 'OTHER'),
     ('Dr', 'person'),
     ('Mahathir', 'person'),
     ('menasihati', 'OTHER'),
     ('mereka', 'OTHER'),
     ('supaya', 'OTHER'),
     ('berhenti', 'OTHER'),
     ('berehat', 'OTHER'),
     ('dan', 'OTHER'),
     ('tidur', 'OTHER'),
     ('sebentar', 'OTHER'),
     ('sekiranya', 'OTHER'),
     ('mengantuk', 'OTHER'),
     ('ketika', 'OTHER'),
     ('memandu', 'OTHER')]




```python
for i in malaya.get_available_entities_models():
    print('Testing %s model'%(i))
    model = malaya.deep_entities(i)
    print(model.predict(string))
    print()
```

    Testing char model
    [('kuala', 'OTHER'), ('lumpur', 'OTHER'), ('sempena', 'OTHER'), ('sambutan', 'OTHER'), ('aidilfitri', 'OTHER'), ('minggu', 'time'), ('depan', 'OTHER'), ('perdana', 'OTHER'), ('menteri', 'OTHER'), ('tun', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('mohamad', 'person'), ('dan', 'OTHER'), ('menteri', 'OTHER'), ('pengangkutan', 'OTHER'), ('anthony', 'person'), ('loke', 'OTHER'), ('siew', 'OTHER'), ('fook', 'OTHER'), ('menitipkan', 'OTHER'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'OTHER'), ('halaman', 'OTHER'), ('masing-masing', 'OTHER'), ('dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('jabatan', 'OTHER'), ('keselamatan', 'OTHER'), ('jalan', 'OTHER'), ('raya', 'OTHER'), ('jkjr', 'location'), ('itu', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]

    Testing word model
    [('kuala', 'location'), ('lumpur', 'location'), ('sempena', 'OTHER'), ('sambutan', 'OTHER'), ('aidilfitri', 'OTHER'), ('minggu', 'OTHER'), ('depan', 'OTHER'), ('perdana', 'person'), ('menteri', 'person'), ('tun', 'person'), ('dr', 'person'), ('mahathir', 'person'), ('mohamad', 'OTHER'), ('dan', 'OTHER'), ('menteri', 'OTHER'), ('pengangkutan', 'organization'), ('anthony', 'person'), ('loke', 'person'), ('siew', 'person'), ('fook', 'person'), ('menitipkan', 'person'), ('pesanan', 'person'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'OTHER'), ('halaman', 'organization'), ('masing-masing', 'OTHER'), ('dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('jabatan', 'OTHER'), ('keselamatan', 'organization'), ('jalan', 'organization'), ('raya', 'organization'), ('jkjr', 'organization'), ('itu', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]

    Testing concat model
    [('kuala', 'location'), ('lumpur', 'location'), ('sempena', 'OTHER'), ('sambutan', 'OTHER'), ('aidilfitri', 'OTHER'), ('minggu', 'time'), ('depan', 'time'), ('perdana', 'person'), ('menteri', 'person'), ('tun', 'person'), ('dr', 'person'), ('mahathir', 'person'), ('mohamad', 'person'), ('dan', 'OTHER'), ('menteri', 'OTHER'), ('pengangkutan', 'OTHER'), ('anthony', 'person'), ('loke', 'person'), ('siew', 'person'), ('fook', 'person'), ('menitipkan', 'OTHER'), ('pesanan', 'OTHER'), ('khas', 'OTHER'), ('kepada', 'OTHER'), ('orang', 'OTHER'), ('ramai', 'OTHER'), ('yang', 'OTHER'), ('mahu', 'OTHER'), ('pulang', 'OTHER'), ('ke', 'OTHER'), ('kampung', 'OTHER'), ('halaman', 'OTHER'), ('masing-masing', 'OTHER'), ('dalam', 'OTHER'), ('video', 'OTHER'), ('pendek', 'OTHER'), ('terbitan', 'OTHER'), ('jabatan', 'organization'), ('keselamatan', 'organization'), ('jalan', 'organization'), ('raya', 'organization'), ('jkjr', 'organization'), ('itu', 'OTHER'), ('dr', 'person'), ('mahathir', 'person'), ('menasihati', 'OTHER'), ('mereka', 'OTHER'), ('supaya', 'OTHER'), ('berhenti', 'OTHER'), ('berehat', 'OTHER'), ('dan', 'OTHER'), ('tidur', 'OTHER'), ('sebentar', 'OTHER'), ('sekiranya', 'OTHER'), ('mengantuk', 'OTHER'), ('ketika', 'OTHER'), ('memandu', 'OTHER')]
