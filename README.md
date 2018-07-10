# Malaya
NLTK like for bahasa Malaysia, powered by Deep Learning

## Requirements
  * Python >= 3.0
  * Tensorflow >= 1.5
  * NumPy >= 1.0
  * Sklearn >= 0.18
  * Fuzzywuzzy
  * unidecode
  * nltk
  * scipy

## Installation
1. Install dependencies
```bash
# using gpu if you installed CUDA and libcudnn
pip3 install sklearn fuzzywuzzy nltk unidecode numpy scipy python-levenshtein tensorflow-gpu==1.5

# using cpu
pip3 install sklearn fuzzywuzzy nltk unidecode numpy scipy python-levenshtein tensorflow==1.5
```

```python
import nltk
nltk.download('punkt')
```

2. Clone this repository
```bash
git clone https://github.com/DevconX/Malaya
```

3. install using setup.py
```bash
python3 setup.py install
```

## Features
  * Deep learning Named entity recognition
  * Deep learning Part-of-speech tagging
  * Naive Part-of-speech tagging
  * Text Normalizer
  * Naive Stemming
  * Naive Bayes
  * Topic Modelling

You can read [README](https://github.com/DevconX/Malaya/tree/master/training) for supported Entities and POS.
## Example

#### check more in example/deep-learning.ipynb
```python
import malaya

# default is 'concat'
model=malaya.deep_learning('char')
model.predict('KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.')

[('KUALA', 'LOC', 'KN'),
 ('LUMPUR:', 'LOC', 'KN'),
 ('Sempena', 'O', 'KN'),
 ('sambutan', 'O', 'KN'),
 ('Aidilfitri', 'EVENT', 'KN'),
 ('minggu', 'O', 'KN'),
 ('depan,', 'O', 'KN'),
 ('Perdana', 'PRN', 'KN'),
 ('Menteri', 'PRN', 'KN'),
 ('Tun', 'PRN', 'KN'),
 ('Dr', 'PRN', 'KN'),
 ('Mahathir', 'PRN', 'KN'),
 ('Mohamad', 'PRN', 'KN'),
```

#### Check more in example/example.ipynb

```python
import malaya

print(malaya.stemming('makanan'))
makan

print(malaya.naive_POS_string('bapa sedang menulis surat'))
[('KN', 'bapa'), ('KETERANGAN', 'sedang'), ('KJ', 'menulis'), ('KN', 'surat')]

print(malaya.naive_POS_string('Dalam ayat pertama Kucing ialah pembuat, menangkap ialah perbuatan melampau dan seekor burung ialah benda yang kena buat atau penyambut'))
[('KS', 'dalam'),
 ('KN', 'ayat'),
 ('KJ', 'pertama'),
 ('KN', 'kucing'),
 ('KPEMERI', 'ialah'),
 ('KJ', 'pembuat'),
 ('', ','),
 ('KJ', 'menangkap'),
 ('KPEMERI', 'ialah'),
 ('KJ', 'perbuatan'),
 ('KJ', 'melampau'),
 ('KH', 'dan'),
 ('KJ', 'seekor'),
 ('KN', 'burung'),
 ('KPEMERI', 'ialah'),
 ('KJ', 'benda'),
 ('KETERANGAN', 'yang'),
 ('KN', 'kena'),
 ('KN', 'buat'),
 ('KH', 'atau'),
 ('KJ', 'penyambut')]

malaya.to_cardinal(123456789)
'seratus dua puluh tiga juta empat ratus lima puluh enam ribu tujuh ratus lapan puluh sembilan'

malaya.to_ordinal(11)
'kesebelas'

bayes=malaya.train_bayes(sample_corpus)
bayes.predict('kelulusan perlu melalui jawatankuasa')
negative: 0.589947
positive: 0.410053

corpus_normalize = ['maka','serious','yeke','masing-masing']
normalizer = malaya.train_normalize(corpus_normalize)
normalizer.normalize('masing2')
'masing-masing'
```

#### Semantic search example, semantic-example/example-semantic-search.ipynb
```python
vectorized = malaya.train_vector(corpus,10)
epoch: 1000, loss: 4.156271
epoch: 2000, loss: 3.501364
epoch: 3000, loss: 2.607565
epoch: 4000, loss: 2.888038
epoch: 5000, loss: 2.324599
epoch: 6000, loss: 2.288272
epoch: 7000, loss: 1.827932
epoch: 8000, loss: 2.251363
epoch: 9000, loss: 1.611805
epoch: 10000, loss: 1.972360
done train

vectorized.semantic_search('mahathir')
[(0, 'mahathir'),
 (1, 'ketokohan'),
 (2, 'berfikiran'),
 (3, 'guru'),
 (4, 'tnb'),
 (5, 'menyebut'),
 (6, 'muda'),
 (7, 'bermasalah'),
 (8, 'mengharapkan'),
 (9, 'hasil')]

vectorized.semantic_search('najib')
[(0, 'najib'),
 (1, 'dijadikan'),
 (2, 'mengatasi'),
 (3, 'tersedia'),
 (4, 'sampai'),
 (5, 'mencari'),
 (6, 'pucuk'),
 (7, 'restoran'),
 (8, 'stres'),
 (9, 'terhasil')]
```

#### k-topic modelling example, topic-example/example-topic.ipynb
```python
lda = malaya.train_lda(corpus,10,cleaning=clearstring_stopwords)
lda.get_topics(10)
[(0, 'projek hutang mengambil tutup bayar memerlukan tanah sewa bulan undi'),
 (1, 'perniagaan malaysia rakyat raya pilihan masalah kali takut bekerjasama penting'),
 (2, 'asli rasa menteri memastikan berjaya pembangunan perdana berkembang masyarakat berjalan'),
 (3, 'menteri low membuat serahkan jho keputusan kapal ph melakukan kena'),
 (4, 'harus mana diterjemahkan wang dasar bank awam bahasa ilmu dana'),
 (5, 'kelulusan ros membenarkan digunakan harga bersedia malaysia seri pesawat airasia'),
 (6, 'rakyat kenyataan negara tersebut kepimpinan islam memudahkan negeri mencari menulis'),
 (7, 'malaysia negara kewangan asli hutang diselesaikan pengalaman pembangunan isu menunjukkan'),
 (8, 'kerajaan bahasa syarikat projek terus dilakukan negara awam swasta tumpuan'),
 (9, 'kementerian memberikan putrajaya kedudukan pengurusan pihak rakyat saiful anak perlembagaan')]

nmf=malaya.train_nmf(corpus,10)
nmf.get_topics(10)
[(0, 'negara malaysia pengalaman ekonomi teknikal membangun tindakan kepentingan kerjasama menunjukkan'),
 (1, 'menteri perdana jemaah seri berlaku razak penjelasan najib baik kuok'),
 (2, 'rakyat kepimpinan malaysia memudahkan hal negeri serius kepentingan berdasarkan pendapatan'),
 (3, 'ros kebenaran pemilihan umno pendaftaran kelulusan perlembagaan minta melebihi tempoh'),
 (4, 'kerajaan sedar pas memastikan mengambil menjatuhkan khususnya pengajaran kemajuan terus'),
 (5, 'kapal jho low dirampas doj perniagaan indonesia anak sivil tuntutan'),
 (6, 'bulan undi harapan pakatan umno diberikan keluarga kasih melonjak terima'),
 (7, 'raya pilihan kononnya semakin penting meminta awal kuok perlembagaan kerusi'),
 (8, 'diselesaikan kewangan hutang tempoh pendek bergantung memerlukan mewujudkan rasa perancangan'),
 (9, 'asli masyarakat jakoa pendapatan temiar harus sumber memalukan arus pelopor')]

malaya.train_lsa(corpus,10)
malaya.topic_lsa(10)
[(0, 'rakyat negara malaysia kerajaan pengalaman menunjukkan kepimpinan isu menteri terus'),
 (1, 'rakyat kepimpinan memudahkan hal serius malaysia negeri pendapatan maklumat kepentingan'),
 (2, 'negara pengalaman malaysia teknikal bidang membangun ekonomi kewangan pendidikan tindakan'),
 (3, 'bulan umno ros undi pemilihan negara harapan keputusan status melebihi'),
 (4, 'jho kapal low kerajaan perniagaan doj pihak sivil dirampas dana'),
 (5, 'kerajaan pas terus masalah memastikan dilakukan masyarakat sedar khususnya proses'),
 (6, 'bulan harapan undi raya pilihan pakatan semakin wang luar bandar'),
 (7, 'raya pilihan penting asli ros semakin pendaftaran masyarakat kononnya ph'),
 (8, 'hutang pendapatan asli diselesaikan kewangan projek masyarakat besar sumber tempoh'),
 (9, 'raya pilihan kewangan bulan diselesaikan hutang negara kerajaan tindakan undi')]
```

## To do

- [x] tokenizer
- [x] stemming
- [x] tatabahasa gathering
- [x] Multinomial naive bayes
- [x] Part-Of-Speech using Regex
- [x] Part-Of-Speech using Deep Learning
- [x] number to words (currency, numbers, position, year)
- [x] stopwords
- [x] normalizer
- [x] top-k hot topic (LSA, NMF, LDA)
- [x] semantic and similar words searching
- [ ] deep learning topic modelling
- [ ] pretrained deep learning sentiment analysis
- [ ] bahasa malaysia checker

## Warning

Install this library with fully aware that this library is still in alpha stage.

Documentation will be released soon.

## Disclaimer

Most of the data gathered using crawlers crawled through targeted malaysia websites. I am not aware of any data protection.

## Authors

* **Husein Zolkepli** - *Initial work* - [huseinzol05](https://github.com/huseinzol05)
