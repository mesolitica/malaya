

```python
import malaya
from malaya import text_functions
```


```python
string = 'KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.'
```


```python
entities = malaya.deep_entities('concat')
pos = malaya.deep_pos('concat')
```

    1.00MB [00:00, 5.49MB/s]                   
    3.00MB [00:00, 3.16MB/s]                          
    1.00MB [00:00, 1.83MB/s]                   
    6.00MB [00:01, 3.01MB/s]                          


    downloading ENTITIES concat settings
    downloading ENTITIES frozen concat model
    downloading POS concat settings
    downloading POS frozen concat model



```python
result_entities = entities.predict(string)
result_entities[:10]
```




    [('kuala', 'location'),
     ('lumpur', 'location'),
     ('sempena', 'OTHER'),
     ('sambutan', 'OTHER'),
     ('aidilfitri', 'OTHER'),
     ('minggu', 'time'),
     ('depan', 'time'),
     ('perdana', 'person'),
     ('menteri', 'person'),
     ('tun', 'person')]




```python
result_pos = pos.predict(string)
result_pos[:10]
```




    [('kuala', 'PROPN'),
     ('lumpur', 'PROPN'),
     ('sempena', 'PROPN'),
     ('sambutan', 'NOUN'),
     ('aidilfitri', 'PROPN'),
     ('minggu', 'PROPN'),
     ('depan', 'ADJ'),
     ('perdana', 'PROPN'),
     ('menteri', 'NOUN'),
     ('tun', 'PROPN')]




```python
text_functions.cluster_pos(result_pos)
```




    {'ADJ': ['depan', 'khas', 'ramai', 'masing-masing', 'pendek', 'mengantuk'],
     'ADP': ['kepada', 'ke'],
     'ADV': ['mahu', 'dalam', 'sebentar'],
     'ADX': [],
     'CCONJ': ['dan'],
     'DET': ['itu'],
     'NOUN': ['sambutan',
      'menteri',
      'pesanan',
      'orang',
      'kampung halaman',
      'video',
      'terbitan jabatan',
      'menasihati',
      'tidur',
      'sekiranya'],
     'NUM': [],
     'PART': [],
     'PRON': ['yang', 'mereka'],
     'PROPN': ['kuala lumpur sempena',
      'aidilfitri minggu',
      'perdana',
      'tun dr mahathir mohamad',
      'pengangkutan anthony loke siew fook menitipkan',
      'keselamatan jalan raya jkjr',
      'dr mahathir',
      'berehat'],
     'SCONJ': ['supaya', 'ketika'],
     'SYM': [],
     'VERB': ['pulang', 'berhenti'],
     'X': []}




```python
text_functions.cluster_entities(result_entities)
```




    {'OTHER': ['sempena sambutan aidilfitri',
      'dan menteri pengangkutan',
      'menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing dalam video pendek terbitan',
      'itu'],
     'law': [],
     'location': ['kuala lumpur'],
     'organization': ['jabatan keselamatan jalan raya jkjr'],
     'person': ['perdana menteri tun dr mahathir mohamad',
      'anthony loke siew fook',
      'dr mahathir'],
     'quantity': [],
     'time': ['minggu depan']}




```python
ngram = text_functions.generate_ngram(result_pos,result_entities)
ngram
```




    ['raya jkjr dr',
     'mahathir mohamad',
     'perdana menteri',
     'minggu depan perdana',
     'jabatan keselamatan jalan',
     'sempena',
     'pulang kampung',
     'menitipkan pesanan orang',
     'dr',
     'menitipkan',
     'menasihati',
     'tidur sekiranya',
     'loke siew',
     'kuala',
     'mahathir',
     'pesanan',
     'perdana',
     'sempena sambutan',
     'kampung halaman video',
     'menteri tun',
     'pulang kampung halaman',
     'perdana menteri tun',
     'menasihati berhenti berehat',
     'mahathir mohamad menteri',
     'anthony loke',
     'tun',
     'berhenti berehat tidur',
     'depan perdana menteri',
     'berehat',
     'jkjr dr mahathir',
     'terbitan jabatan',
     'raya jkjr',
     'mohamad',
     'video terbitan',
     'sekiranya',
     'sekiranya memandu',
     'orang pulang kampung',
     'loke',
     'fook menitipkan',
     'orang pulang',
     'lumpur sempena',
     'loke siew fook',
     'halaman video terbitan',
     'berehat tidur sekiranya',
     'kampung',
     'lumpur sempena sambutan',
     'sambutan aidilfitri',
     'menitipkan pesanan',
     'menteri pengangkutan anthony',
     'video terbitan jabatan',
     'memandu',
     'siew',
     'mohamad menteri pengangkutan',
     'keselamatan',
     'siew fook',
     'anthony loke siew',
     'dr mahathir',
     'berhenti',
     'jalan raya',
     'aidilfitri minggu',
     'berehat tidur',
     'fook menitipkan pesanan',
     'pesanan orang pulang',
     'orang',
     'halaman',
     'dr mahathir mohamad',
     'sambutan aidilfitri minggu',
     'kuala lumpur',
     'dr mahathir menasihati',
     'aidilfitri',
     'fook',
     'pengangkutan',
     'video',
     'menteri',
     'lumpur',
     'mohamad menteri',
     'jabatan keselamatan',
     'kampung halaman',
     'jabatan',
     'pulang',
     'tidur',
     'pesanan orang',
     'mahathir menasihati',
     'jalan raya jkjr',
     'pengangkutan anthony loke',
     'sambutan',
     'menasihati berhenti',
     'menteri tun dr',
     'kuala lumpur sempena',
     'mahathir menasihati berhenti',
     'depan',
     'terbitan jabatan keselamatan',
     'jalan',
     'pengangkutan anthony',
     'sempena sambutan aidilfitri',
     'jkjr',
     'anthony',
     'depan perdana',
     'keselamatan jalan raya',
     'jkjr dr',
     'aidilfitri minggu depan',
     'siew fook menitipkan',
     'terbitan',
     'tun dr mahathir',
     'raya',
     'halaman video',
     'minggu',
     'berhenti berehat',
     'minggu depan',
     'keselamatan jalan',
     'tidur sekiranya memandu',
     'menteri pengangkutan',
     'tun dr']




```python
text_functions.cluster_words(ngram)
```




    ['raya jkjr dr',
     'menteri tun dr',
     'dr mahathir mohamad',
     'sambutan aidilfitri minggu',
     'minggu depan perdana',
     'mahathir menasihati berhenti',
     'kuala lumpur sempena',
     'dr mahathir menasihati',
     'jabatan keselamatan jalan',
     'orang pulang kampung',
     'loke siew fook',
     'halaman video terbitan',
     'menitipkan pesanan orang',
     'terbitan jabatan keselamatan',
     'berehat tidur sekiranya',
     'sempena sambutan aidilfitri',
     'kampung halaman video',
     'lumpur sempena sambutan',
     'menteri pengangkutan anthony',
     'video terbitan jabatan',
     'keselamatan jalan raya',
     'pulang kampung halaman',
     'perdana menteri tun',
     'aidilfitri minggu depan',
     'menasihati berhenti berehat',
     'mahathir mohamad menteri',
     'mohamad menteri pengangkutan',
     'siew fook menitipkan',
     'anthony loke siew',
     'berhenti berehat tidur',
     'depan perdana menteri',
     'tun dr mahathir',
     'jalan raya jkjr',
     'jkjr dr mahathir',
     'pengangkutan anthony loke',
     'fook menitipkan pesanan',
     'tidur sekiranya memandu',
     'pesanan orang pulang']




```python

```
