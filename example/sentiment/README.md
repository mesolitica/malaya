

```python
import malaya
import pandas as pd
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
bayes_sentiment = malaya.pretrained_bayes_sentiment()
```


```python
positive_text = 'Kerajaan negeri Kelantan mempersoalkan motif kenyataan Menteri Kewangan Lim Guan Eng yang hanya menyebut Kelantan penerima terbesar bantuan kewangan dari Kerajaan Persekutuan. Sedangkan menurut Timbalan Menteri Besarnya, Datuk Mohd Amar Nik Abdullah, negeri lain yang lebih maju dari Kelantan turut mendapat pembiayaan dan pinjaman.'
```


```python
bayes_sentiment.predict(positive_text)
```




    'positive'




```python
bayes_sentiment.predict(positive_text,get_proba=True)
```




    {'negative': 0.45994922524730797, 'positive': 0.5400507747526919}




```python
negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
bayes_sentiment.predict(negative_text,get_proba=True)
```




    {'negative': 0.45994922524730797, 'positive': 0.5400507747526919}




```python
bayes_sentiment.predict_batch([negative_text,negative_text],get_proba=True)
```




    [{'negative': 0.45994922524730797, 'positive': 0.5400507747526919},
     {'negative': 0.45994922524730797, 'positive': 0.5400507747526919}]




```python
xgb_sentiment = malaya.pretrained_xgb_sentiment()
xgb_sentiment.predict(negative_text,get_proba=True)
```




    {'negative': 0.4963528, 'positive': 0.5036472}




```python
sentiment_available_models = malaya.get_available_sentiment_models()
sentiment_available_models
```




    ['bahdanau', 'hierarchical', 'luong', 'bidirectional', 'fast-text', 'stack']




```python
for i in sentiment_available_models:
    print('Testing %s model'%(i))
    news_sentiment = malaya.deep_sentiment(i)
    print(news_sentiment.predict(negative_text))
    print()
```

    Testing bahdanau model
    {'negative': 0.9987398, 'positive': 0.0012602198, 'attention': [['kerajaan', 0.04892652], ['sebenarnya', 0.020006381], ['sangat', 0.017095787], ['bencikan', 0.016283441], ['rakyatnya', 0.019184407], ['minyak', 0.0450745], ['naik', 0.019356105], ['dan', 0.019716889], ['segalanya', 0.794356]]}

    Testing hierarchical model
    {'negative': 0.15909557, 'positive': 0.8409045, 'attention': [['kerajaan', 0.0019194365], ['sebenarnya', 0.004214599], ['sangat', 0.028645746], ['bencikan', 0.040212832], ['rakyatnya', 0.111732095], ['minyak', 0.14156568], ['naik', 0.24453603], ['dan', 0.24232633], ['segalanya', 0.1848472]]}

    Testing luong model
    {'negative': 0.9172633, 'positive': 0.08273667, 'attention': [['kerajaan', 0.11111111], ['sebenarnya', 0.11111111], ['sangat', 0.11111111], ['bencikan', 0.11111111], ['rakyatnya', 0.11111111], ['minyak', 0.11111111], ['naik', 0.11111111], ['dan', 0.11111111], ['segalanya', 0.11111111]]}

    Testing bidirectional model
    {'negative': 0.9765419, 'positive': 0.023458067}

    Testing fast-text model
    {'negative': 0.7411294, 'positive': 0.25887057}

    Testing stack model
    downloading SENTIMENT frozen stack model


    65.0MB [03:12, 2.97s/MB]                          


    downloading SENTIMENT stack dictionary


    1.00MB [00:01, 1.38s/MB]                   


    {'negative': 0.636537, 'positive': 0.36346304}




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
        Neutral       1.00      0.13      0.24        15
       Positive       0.63      1.00      0.77        37

    avg / total       0.63      0.64      0.53        61



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

         adidas       0.97      0.62      0.75       310
          apple       0.96      0.60      0.74       419
         hungry       0.82      0.92      0.87      1070
       kerajaan       0.84      0.82      0.83      1371
           nike       0.93      0.60      0.73       326
    pembangkang       0.72      0.85      0.78      1563

    avg / total       0.82      0.81      0.80      5059




```python
bayes.predict('saya suka kerajaan dan anwar ibrahim')
```




    'pembangkang'
