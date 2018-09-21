

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
embedded = malaya.malaya_word2vec(256)
```


```python
word_vector = malaya.Word2Vec(embedded['nce_weights'], embedded['dictionary'])
```


```python
word = 'anwar'
print("Embedding layer: 8 closest words to: '%s'"%(word))
print(word_vector.n_closest(word=word, num_closest=8, metric='cosine'))
```

    Embedding layer: 8 closest words to: 'anwar'
    [['rizal', 0.4755251407623291], ['johari', 0.44806742668151855], ['muhyiddin', 0.4442579746246338], ['rafizi', 0.442638635635376], ['shamsul', 0.43479591608047485], ['azalina', 0.4314958453178406], ['muhammad', 0.4261718988418579], ['isa', 0.4253132939338684]]



```python
print(word_vector.analogy('anwar', 'penjara', 'kerajaan', 5))
```

    ['kerajaan', 'penjara', 'kpm', 'pucuk', 'trump']



```python
word_vector.calculator('anwar + amerika + mahathir', num_closest=8, metric='cosine',
                      return_similarity=False)
```




    ['mahathir',
     'anwar',
     'amerika',
     'subramaniam',
     'jinping',
     'obama',
     'mahkota',
     'thomas',
     'taufek']




```python
word_vector.calculator('anwar * amerika', num_closest=8, metric='cosine',
                      return_similarity=False)
```




    ['mana',
     'apa',
     'kritikal',
     'ini',
     'penceramah',
     'institusi',
     'nasional',
     'berkepentingan',
     'paling']




```python

```
