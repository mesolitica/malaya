

```python
import malaya
```

    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
news_sentiment = malaya.pretrained_bayes_sentiment()
```


```python
positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
```


```python
news_sentiment.predict(positive_text)
```




    [('negative', 0.18390337653291502), ('positive', 0.8160966234670884)]




```python
negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
```


```python
news_sentiment.predict(negative_text)
```




    [('negative', 0.21693467273378716), ('positive', 0.7830653272662134)]




```python
sentiment_available_models = malaya.get_available_sentiment_models()
sentiment_available_models
```




    ['bahdanau', 'attention', 'luong', 'normal']




```python
for i in sentiment_available_models:
    print('Testing %s model'%(i))
    news_sentiment = malaya.deep_sentiment(i)
    print(news_sentiment.predict(negative_text))
    print()
```

    Testing bahdanau model
    downloading word2vec-256 embedded

    downloading frozen bahdanau model

    {'negative': 0.4206314, 'positive': 0.5793686, 'attention': [['kerajaan', 0.07571377], ['bencikan', 0.2937163], ['rakyatnya', 0.14874507], ['minyak', 0.48182485]]}

    Testing attention model
    downloading frozen attention model

    {'negative': 0.45025843, 'positive': 0.54974157, 'attention': [['kerajaan', 0.26210415], ['bencikan', 0.3908129], ['rakyatnya', 0.27151617], ['minyak', 0.07556677]]}

    Testing luong model
    downloading frozen luong model

    {'negative': 0.3331387, 'positive': 0.66686136, 'attention': [['kerajaan', 0.038200114], ['bencikan', 0.8273579], ['rakyatnya', 0.09624269], ['minyak', 0.038199265]]}

    Testing normal model
    downloading frozen normal model

    {'negative': 0.560395, 'positive': 0.439605}




```python

```
