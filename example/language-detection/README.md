

```python
import malaya
```

    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
malaya.get_language_labels()
```




    {0: 'OTHER', 1: 'ENGLISH', 2: 'INDONESIA', 3: 'MALAY'}




```python
chinese_text = '今天是６月１８号，也是Muiriel的生日！'
english_text = 'i totally love it man'
indon_text = 'berbicara dalam bahasa Indonesia membutuhkan teknologi yang baik untuk bekerja dengan baik, tetapi teknologi yang sulit didapat'
malay_text = 'beliau berkata program Inisitif Peduli Rakyat (IPR) yang diperkenalkan oleh kerajaan negeri Selangor lebih besar sumbangannya'
```


```python
malaya.detect_language(chinese_text)
```

    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator CountVectorizer from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /usr/local/lib/python3.6/site-packages/sklearn/base.py:311: UserWarning: Trying to unpickle estimator MultinomialNB from version 0.19.1 when using version 0.19.2. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)





    'OTHER'




```python
malaya.detect_language(english_text)
```




    'ENGLISH'




```python
malaya.detect_language(indon_text)
```




    'MALAY'




```python
malaya.detect_language(malay_text)
```




    'MALAY'




```python
malaya.detect_language(malay_text,get_proba=True)
```




    {'OTHER': 0.0,
     'ENGLISH': 0.0,
     'INDONESIA': 1.485952831042105e-173,
     'MALAY': 1.0}




```python
malaya.detect_languages([english_text,malay_text])
```




    ['ENGLISH', 'MALAY']




```python
import malaya
malaya.detect_languages([english_text,malay_text],get_proba=True)
```




    [{'OTHER': 1.2628201695336024e-54,
      'ENGLISH': 1.0,
      'INDONESIA': 8.393192343031928e-59,
      'MALAY': 7.860504737854806e-51},
     {'OTHER': 0.0,
      'ENGLISH': 0.0,
      'INDONESIA': 1.485952831042105e-173,
      'MALAY': 1.0}]




```python

```
