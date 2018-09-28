

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
malaya.sastrawi_stemmer('saya tengah berjalan')
```




    'saya tengah jalan'




```python
malaya.sastrawi_stemmer('saya tengah berjalankan sangat-sangat')
```




    'saya tengah jalan sangat'




```python
malaya.sastrawi_stemmer('menarik')
```




    'tarik'




```python
stemmer = malaya.deep_stemmer()
```


```python
stemmer.stem('saya tengah berjalankan sangat-sangat')
```




    'saya tengah jalan sangat'




```python
stemmer.stem('saya sangat sukakan awak')
```




    'saya sangat suka awak'




```python
stemmer.stem('saya sangat suakkan awak')
```




    'saya sangat suak awak'
