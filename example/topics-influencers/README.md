

```python
import malaya
```

    Using TensorFlow backend.



```python
news = 'najib razak dan mahathir mengalami masalah air di kemamam terengganu'
```


```python
malaya.fuzzy_get_topics(news)
```




    ['masalah air', 'mahathir', 'najib razak']




```python
malaya.fuzzy_get_influencers(news)
```




    ['mahathir', 'najib razak']




```python
second_news = 'ikat penyedia perkhidmatan jalur lebar Telekom Malaysia (TM) perlu mencari jalan penyelesaian bagi meningkatkan akses capaian Internet ke seluruh negara, kata Menteri Komunikasi dan Multimedia, Gobind Singh Deo. Beliau berkata menjadi dasar kerajaan untuk membekalkan akses Internet jalur lebar kepada semua dan memberi penekanan kepada kualiti perkhidmatan yang terbaik. "Dasar kerajaan untuk bekalkan akses kepada semua bukan sekadar pembekalan sahaja tetapi beri penekanan kepada kualiti perkhidmatan yang baik dan dapat bersaing dengan negara lain pada tahap antarabangsa," kata Gobind Singh menerusi catatan di laman rasmi Twitter beliau, malam tadi. Beliau berkata demikian sebagai respons terhadap aduan beberapa pengguna Twitter berhubung akses Internet yang masih tidak stabil serta harga yang tidak berpatutan di beberapa lokasi di seluruh negara.'
```


```python
malaya.fuzzy_get_topics(second_news)
```




    ['teknologi',
     'telekom malaysia',
     'kerajaan',
     'politik',
     'perkhidmatan awam',
     'internet',
     'pendidikan',
     'twitter',
     'kkmm',
     'sosial media']




```python
malaya.fuzzy_get_influencers(second_news)
```




    ['gobind singh deo']




```python
topics_similarity = malaya.fast_get_topics()
```


```python
topics_similarity.get_similarity(news)
```




    ['mahathir', 'najib razak', 'tan sri mokhzani mahathir']




```python
influencers_similarity = malaya.fast_get_influencers()
```


```python
influencers_similarity.get_similarity(news)
```




    ['mahathir', 'najib razak', 'tan sri mokhzani mahathir']




```python
influencers_similarity.get_similarity(second_news)
```




    ['mic',
     'gobind singh deo',
     'parti amanah',
     'majlis pakatan harapan',
     'ppbm',
     'pakatan harapan',
     'parti pribumi bersatu malaysia',
     'jabatan perancangan bandar dan desa',
     'parti islam semalaysia']




```python
deep_topic = malaya.deep_get_topics()
```

    minibatch loop: 100%|██████████| 157/157 [00:18<00:00,  8.87it/s, cost=1.75]
    minibatch loop: 100%|██████████| 157/157 [00:17<00:00,  9.12it/s, cost=0.141]
    minibatch loop: 100%|██████████| 157/157 [00:17<00:00,  9.02it/s, cost=0.0708]
    minibatch loop: 100%|██████████| 157/157 [00:17<00:00,  9.13it/s, cost=0.0447]
    minibatch loop: 100%|██████████| 157/157 [00:17<00:00,  9.15it/s, cost=0.0469]



```python
deep_topic.get_similarity(second_news, anchor = 0.5)
```




    ['mikro-ekonomi',
     'pusat daerah mangundi',
     'k-pop',
     'datuk seri ti lian ker',
     'keluar parti',
     'pilihan raya umum ke-14',
     'malaysia-indonesia',
     'tunku ismail idris',
     'parti pribumi bersatu malaysia',
     'makro-ekonomi',
     'kkmm',
     '#fakenews',
     'datuk seri azmin ali']




```python
deep_influencer = malaya.deep_get_influencers()
```

    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  9.06it/s, cost=3.79]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  9.16it/s, cost=1.95]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  9.03it/s, cost=0.947]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  9.16it/s, cost=0.535]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  9.09it/s, cost=0.463]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  9.03it/s, cost=0.336]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  8.98it/s, cost=0.334]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  9.04it/s, cost=0.285]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  8.84it/s, cost=0.309]
    minibatch loop: 100%|██████████| 20/20 [00:02<00:00,  9.21it/s, cost=0.377]



```python
deep_influencer.get_similarity(news, anchor = 0.5)
```




    ['mahathir', 'najib razak']




```python
deep_influencer.get_similarity(second_news, anchor = 0.5)
```




    ['gobind singh deo']
