

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
malays = malaya.load_malay_dictionary()
corrector = malaya.naive_speller(malays)
```


```python
corrector.correct('mknn')
```

    [('mena', 50), ('makin', 67), ('makan', 67), ('makanan', 73), ('mini', 50), ('min', 57), ('main', 50), ('makna', 67), ('mani', 50), ('mana', 50), ('maun', 50), ('menu', 50)]





    'makanan'




```python
corrector.correct('tmpat',debug=True)
```

    [('tumpat', 91), ('tepat', 80), ('tempat', 91)]





    'tumpat'




```python
%time
corrector.correct('mknn',first_char=True)
```

    CPU times: user 4 µs, sys: 1e+03 ns, total: 5 µs
    Wall time: 7.87 µs
    [('mena', 50), ('makin', 67), ('makan', 67), ('makanan', 73), ('mini', 50), ('min', 57), ('main', 50), ('makna', 67), ('mani', 50), ('mana', 50), ('maun', 50), ('menu', 50)]





    'makanan'




```python
%time
corrector.correct('mknn',first_char=False)
```

    CPU times: user 4 µs, sys: 1 µs, total: 5 µs
    Wall time: 8.82 µs
    [('mena', 50), ('makin', 67), ('makan', 67), ('makanan', 73), ('ikan', 50), ('mini', 50), ('min', 57), ('ikon', 50), ('makna', 67), ('main', 50), ('mani', 50), ('ken', 57), ('mana', 50), ('kan', 57), ('kun', 57), ('kon', 57), ('menu', 50), ('akan', 50), ('maun', 50)]





    'makanan'




```python
corrector.correct('tempat')
```

    [('tempat', 100)]





    'tempat'




```python

```
