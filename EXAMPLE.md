# Example

#### check more in example/deep-learning.ipynb
```python
import malaya

# default is 'attention'
# support 'attention','concat','char'
# in term of accuracy (accurate to not)
# attention > concat > char
# in term of speed (fastest to slowest)
# char > concat > attention
model=malaya.deep_learning()
model.predict('KUALA LUMPUR: Sempena sambutan Aidilfitri minggu depan, Perdana Menteri Tun Dr Mahathir Mohamad dan Menteri Pengangkutan Anthony Loke Siew Fook menitipkan pesanan khas kepada orang ramai yang mahu pulang ke kampung halaman masing-masing. Dalam video pendek terbitan Jabatan Keselamatan Jalan Raya (JKJR) itu, Dr Mahathir menasihati mereka supaya berhenti berehat dan tidur sebentar  sekiranya mengantuk ketika memandu.')

KUALA LOC KN
LUMPUR: LOC KN
Sempena O KN
sambutan O KN
Aidilfitri EVENT KN
minggu TIME KN
depan, TIME KN
Perdana PRN KN
Menteri PRN KN
Tun PRN KN
Dr PRN KN
Mahathir PRN KN
Mohamad PRN KN
dan O KH
Menteri NORP KN
Pengangkutan NORP KN
Anthony PRN KN
Loke PRN KN
Siew PRN KN
Fook PRN KN
menitipkan NORP KN
pesanan NORP KN
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

#### word2vec example, example/load-embedded.ipynb
```python
embedded = malaya.get_word2vec(64)
word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
word = 'mahathir'
print("Embedding layer: 8 closest words to: '%s'"%(word))
print(word_vector.n_closest(word=word, num_closest=8, metric='cosine'))
[['azalina', 0.42232489585876465], ['kyi', 0.39500147104263306], ['jho', 0.39347755908966064], ['kandis', 0.39313936233520508], ['kejahilan', 0.38966810703277588], ['razali', 0.38890910148620605], ['hishamuddin', 0.38873559236526489], ['tia', 0.38481009006500244]]
print(word_vector.analogy('anwar', 'penjara', 'kerajaan', 5))
['kerajaan', 'lima', 'tujuh', 'enam', 'hadapan']
```

#### k-topic modelling example, example/example-topic.ipynb
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
