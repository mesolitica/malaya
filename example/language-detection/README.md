

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
multinomial = malaya.multinomial_detect_languages()
multinomial.predict(chinese_text)
```




    'OTHER'




```python
multinomial.predict(english_text)
```




    'ENGLISH'




```python
multinomial.predict(indon_text)
```




    'MALAY'




```python
multinomial.predict(malay_text)
```




    'MALAY'




```python
multinomial.predict(malay_text,get_proba=True)
```




    {'OTHER': 0.0,
     'ENGLISH': 0.0,
     'INDONESIA': 1.485952831042105e-173,
     'MALAY': 1.0}




```python
multinomial.predict_batch([english_text,malay_text])
```




    ['ENGLISH', 'MALAY']




```python
multinomial.predict_batch([english_text,malay_text],get_proba=True)
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
xgb = malaya.xgb_detect_languages()
xgb.predict(chinese_text)
```




    'OTHER'




```python
xgb.predict(indon_text,get_proba=True)
```




    {'OTHER': 6.92337e-10,
     'ENGLISH': 3.507782e-11,
     'INDONESIA': 0.9995041,
     'MALAY': 0.0004959471}




```python
xgb.predict_batch([indon_text,malay_text],get_proba=True)
```




    [{'OTHER': 6.92337e-10,
      'ENGLISH': 3.507782e-11,
      'INDONESIA': 0.9995041,
      'MALAY': 0.0004959471},
     {'OTHER': 1.174448e-09,
      'ENGLISH': 1.4715874e-10,
      'INDONESIA': 0.001421933,
      'MALAY': 0.9985781}]




```python

```
