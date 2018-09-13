

```python
import malaya
```

    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
news = 'najib razak dan mahathir mengalami masalah air di kemamam terengganu'
```


```python
malaya.get_topics(news)
```




    ['terengganu', 'najib razak', 'masalah air', 'mahathir']




```python
malaya.get_influencers(news)
```




    ['najib razak', 'mahathir']




```python
news = 'ikat penyedia perkhidmatan jalur lebar Telekom Malaysia (TM) perlu mencari jalan penyelesaian bagi meningkatkan akses capaian Internet ke seluruh negara, kata Menteri Komunikasi dan Multimedia, Gobind Singh Deo. Beliau berkata menjadi dasar kerajaan untuk membekalkan akses Internet jalur lebar kepada semua dan memberi penekanan kepada kualiti perkhidmatan yang terbaik. "Dasar kerajaan untuk bekalkan akses kepada semua bukan sekadar pembekalan sahaja tetapi beri penekanan kepada kualiti perkhidmatan yang baik dan dapat bersaing dengan negara lain pada tahap antarabangsa," kata Gobind Singh menerusi catatan di laman rasmi Twitter beliau, malam tadi. Beliau berkata demikian sebagai respons terhadap aduan beberapa pengguna Twitter berhubung akses Internet yang masih tidak stabil serta harga yang tidak berpatutan di beberapa lokasi di seluruh negara.'
```


```python
malaya.get_topics(news)
```




    ['telekom malaysia',
     'kerajaan',
     'saudi arabia',
     'politik',
     'teknologi',
     'sosial media',
     'internet',
     'kaum melayu',
     'twitter']




```python
malaya.get_influencers(news)
```




    ['gobind singh deo']




```python

```
