

```python
import malaya
import pandas as pd
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




    'positive'




```python
news_sentiment.predict(positive_text,get_proba=True)
```




    {'negative': 0.2341921505296387, 'positive': 0.7658078494703627}




```python
negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
news_sentiment.predict(negative_text,get_proba=True)
```




    {'negative': 0.19026157818306375, 'positive': 0.8097384218169362}




```python
news_sentiment.predict_batch([negative_text,negative_text],get_proba=True)
```




    [{'negative': 0.19026157818306375, 'positive': 0.8097384218169362},
     {'negative': 0.19026157818306375, 'positive': 0.8097384218169362}]




```python
news_sentiment = malaya.pretrained_xgb_sentiment()
news_sentiment.predict(negative_text,get_proba=True)
```

      0%|          | 0/2.806271553039551 [00:00<?, ?MB/s]

    downloading pickled tfidf vectorizations


    3MB [00:02,  1.45MB/s]                                       





    {'negative': 0.38166726, 'positive': 0.61833274}




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
    {'negative': 0.42894447, 'positive': 0.57105553, 'attention': [['kerajaan', 0.07550501], ['bencikan', 0.29057813], ['rakyatnya', 0.1474754], ['minyak', 0.48644146]]}

    Testing attention model
    {'negative': 0.43696418, 'positive': 0.56303585, 'attention': [['kerajaan', 0.26913235], ['bencikan', 0.38034844], ['rakyatnya', 0.28445157], ['minyak', 0.06606761]]}

    Testing luong model
    {'negative': 0.49942672, 'positive': 0.5005733, 'attention': [['kerajaan', 0.056799203], ['bencikan', 0.2996163], ['rakyatnya', 0.58679783], ['minyak', 0.056786623]]}

    Testing normal model
    {'negative': 0.48782662, 'positive': 0.5121734}




```python
df = pd.read_csv('tests/02032018.csv',sep=';')
df = df.iloc[3:,1:]
df.columns = ['text','label']
corpus = df.text.tolist()
```


```python
dataset = [[df.iloc[i,0],df.iloc[i,1]] for i in range(df.shape[0])]
bayes=malaya.bayes_sentiment(dataset)
```

                 precision    recall  f1-score   support

       Negative       0.00      0.00      0.00         9
        Neutral       0.67      0.08      0.15        24
       Positive       0.47      0.96      0.63        28

    avg / total       0.48      0.48      0.35        61



    /usr/local/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
bayes.predict(dataset[0][0])
```




    'Positive'




```python
bayes = malaya.bayes_sentiment('tests/local')
```

                 precision    recall  f1-score   support

         adidas       0.97      0.53      0.69       325
          apple       0.98      0.54      0.70       488
         hungry       0.79      0.91      0.85      1038
       kerajaan       0.87      0.80      0.83      1380
           nike       0.93      0.56      0.70       293
    pembangkang       0.69      0.88      0.78      1535

    avg / total       0.82      0.79      0.79      5059




```python
bayes.predict('saya suka kerajaan dan anwar ibrahim')
```




    'pembangkang'
