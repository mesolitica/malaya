

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
malaya.to_cardinal(123456789)
```




    'seratus dua puluh tiga juta empat ratus lima puluh enam ribu tujuh ratus lapan puluh sembilan'




```python
malaya.to_cardinal(-123456789)
```




    'negatif seratus dua puluh tiga juta empat ratus lima puluh enam ribu tujuh ratus lapan puluh sembilan'




```python
malaya.to_cardinal(-1234567.89)
```




    'negatif satu juta dua ratus tiga puluh empat ribu lima ratus enam puluh tujuh perpuluhan lapan sembilan'




```python
malaya.to_ordinal(11)
```




    'kesebelas'




```python

```
