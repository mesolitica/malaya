

```python
import malaya
```

    Using TensorFlow backend.



```python
malays = malaya.load_malay_dictionary()
corrector = malaya.naive_speller(malays)
```


```python
corrector.correct('mknn')
```

    [(('mani', False), 50), (('menu', False), 50), (('main', False), 50), (('maun', False), 50), (('mena', False), 50), (('makan', False), 67), (('makin', False), 67), (('min', False), 57), (('mana', False), 50), (('makna', False), 67), (('mini', False), 50), (('makanan', False), 73)]






    'makanan'




```python
corrector.correct('tmpat',debug=True)
```

    [(('tempat', False), 91), (('tumpat', True), 91), (('tepat', False), 80)]






    'tempat'




```python
%%time
corrector.correct('mknn',first_char=True)
```

    [(('mani', False), 50), (('menu', False), 50), (('main', False), 50), (('maun', False), 50), (('mena', False), 50), (('makan', False), 67), (('makin', False), 67), (('min', False), 57), (('mana', False), 50), (('makna', False), 67), (('mini', False), 50), (('makanan', False), 73)]

    CPU times: user 144 ms, sys: 0 ns, total: 144 ms
    Wall time: 140 ms





    'makanan'




```python
%%time
corrector.correct('mknn',first_char=False)
```

    [(('mani', False), 50), (('menu', False), 50), (('main', False), 50), (('kun', False), 57), (('maun', False), 50), (('mena', False), 50), (('makan', False), 67), (('ikan', False), 50), (('min', False), 57), (('kon', False), 57), (('akan', False), 50), (('mana', False), 50), (('makin', False), 67), (('ken', False), 57), (('makna', False), 67), (('ikon', False), 50), (('mini', False), 50), (('kan', False), 57), (('makanan', False), 73)]

    CPU times: user 216 ms, sys: 0 ns, total: 216 ms
    Wall time: 214 ms





    'makanan'




```python
corrector.correct('tempat')
```

    [(('tempat', False), 100)]






    'tempat'
