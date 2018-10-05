

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
string = 'y u xsuka makan kt situ'
another = 'i mmg xska mknn kampng'
```


```python
malaya.basic_normalizer(string)
```




    'kenapa awak xsuka makan kt situ'




```python
malays = malaya.load_malay_dictionary()
normalizer = malaya.fuzzy_normalizer(malays)
```


```python
normalizer.normalize(string)
```




    'kenapa awak tak suka makan kat situ'




```python
normalizer.normalize(another)
```




    'saya memang tak saka makanan kampung'




```python
normalizer = malaya.spell_normalizer(malays)
```


```python
normalizer.normalize(string,debug=True)
```




    'kenapa awak tak suka makan kat situ'




```python
normalizer.normalize(another,debug=True)
```

    [('saka', 86), ('spa', 67), ('sika', 86), ('sia', 67), ('seka', 86), ('sua', 67), ('ski', 67), ('suka', 86), ('soka', 86)]
    [('mani', 50), ('akan', 50), ('makna', 67), ('mena', 50), ('menu', 50), ('ken', 57), ('mini', 50), ('kon', 57), ('ikan', 50), ('ikon', 50), ('min', 57), ('makanan', 73), ('maun', 50), ('makin', 67), ('main', 50), ('kun', 57), ('makan', 67), ('kan', 57), ('mana', 50)]
    [('kampung', 92)]





    'saya memang tak saka makanan kampung'




```python
normalizer = malaya.deep_normalizer()
```


```python
normalizer.normalize(string)
```




    'eye uau t sakuai makan kati situ'
