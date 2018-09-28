

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

    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator MultinomialNB from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator TfidfTransformer from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator TfidfVectorizer from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)



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




    {'negative': 0.19013070424544617, 'positive': 0.8098692957545561}




```python
negative_text = 'kerajaan sebenarnya sangat bencikan rakyatnya, minyak naik dan segalanya'
bayes_sentiment.predict(negative_text,get_proba=True)
```




    {'negative': 0.3723520311772738, 'positive': 0.6276479688227262}




```python
bayes_sentiment.predict_batch([negative_text,negative_text],get_proba=True)
```




    [{'negative': 0.3723520311772738, 'positive': 0.6276479688227262},
     {'negative': 0.3723520311772738, 'positive': 0.6276479688227262}]




```python
xgb_sentiment = malaya.pretrained_xgb_sentiment()
xgb_sentiment.predict(negative_text,get_proba=True)
```

    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator TfidfTransformer from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator TfidfVectorizer from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)





    {'negative': 0.3550796, 'positive': 0.6449204}




```python
sentiment_available_models = malaya.get_available_sentiment_models()
sentiment_available_models
```




    ['bahdanau', 'hierarchical', 'luong', 'bidirectional', 'fast-text']




```python
for i in sentiment_available_models:
    print('Testing %s model'%(i))
    news_sentiment = malaya.deep_sentiment(i)
    print(news_sentiment.predict(negative_text))
    print()
```

    Testing bahdanau model
    {'negative': 0.99867034, 'positive': 0.0013296733, 'attention': [['kerajaan', 0.04794306], ['sebenarnya', 0.019771717], ['sangat', 0.01688926], ['bencikan', 0.016135536], ['rakyatnya', 0.018904446], ['minyak', 0.044418886], ['naik', 0.01919316], ['dan', 0.019459246], ['segalanya', 0.79728466]]}

    Testing hierarchical model
    {'negative': 0.17229004, 'positive': 0.82771, 'attention': [['kerajaan', 0.0039255945], ['sebenarnya', 0.00531989], ['sangat', 0.0146343], ['bencikan', 0.029050263], ['rakyatnya', 0.073665366], ['minyak', 0.28049424], ['naik', 0.28645015], ['dan', 0.21241833], ['segalanya', 0.09404183]]}

    Testing luong model
    {'negative': 0.9494788, 'positive': 0.05052119, 'attention': [['kerajaan', 0.11111111], ['sebenarnya', 0.11111111], ['sangat', 0.11111111], ['bencikan', 0.11111111], ['rakyatnya', 0.11111111], ['minyak', 0.11111111], ['naik', 0.11111111], ['dan', 0.11111111], ['segalanya', 0.11111111]]}

    Testing bidirectional model
    {'negative': 0.9589319, 'positive': 0.041068066}

    Testing fast-text model
    {'negative': 0.7411294, 'positive': 0.25887057}




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

       Negative       0.00      0.00      0.00        15
        Neutral       0.29      0.17      0.21        12
       Positive       0.63      1.00      0.77        34

    avg / total       0.41      0.59      0.47        61



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

         adidas       0.91      0.59      0.71       297
          apple       0.98      0.63      0.76       471
         hungry       0.83      0.91      0.87      1074
       kerajaan       0.84      0.82      0.83      1387
           nike       0.95      0.58      0.72       321
    pembangkang       0.70      0.85      0.77      1509

    avg / total       0.82      0.80      0.80      5059




```python
bayes.predict('saya suka kerajaan dan anwar ibrahim')
```




    'pembangkang'
