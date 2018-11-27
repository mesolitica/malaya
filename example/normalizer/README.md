

```python
import malaya
```

    Using TensorFlow backend.



```python
string = 'y u xsuka makan HUSEIN kt situ tmpt'
another = 'i mmg xska mknn HUSEIN kampng tempt'
```


```python
malaya.basic_normalizer(string)
```




    'kenapa awak xsuka makan Husein kt situ tmpt'




```python
malays = malaya.load_malay_dictionary()
normalizer = malaya.fuzzy_normalizer(malays)
```

      0%|          | 0.00/0.19 [00:00<?, ?MB/s]

    downloading Malay texts


    1.00MB [00:00, 10.2MB/s]                   



```python
normalizer.normalize(string)
```




    'kenapa awak tak suka makan Husein kat situ tempat'




```python
normalizer.normalize(another)
```




    'saya memang tak saka makanan Husein kampung tempat'




```python
normalizer = malaya.spell_normalizer(malays)
```


```python
normalizer.normalize(string,debug=True)
```

    [(('tat', False), 11), (('ampu', False), 15), (('tapa', False), 10), (('empat', True), 15), (('timpa', False), 20), (('temut', False), 15), (('impi', False), 15), (('tepu', False), 10), (('tumit', False), 20), (('tuit', False), 15), (('amit', False), 21), (('tampi', True), 15), (('tampa', False), 15), (('taut', False), 15), (('umut', False), 21), (('tepet', False), 15), (('tumpu', False), 20), (('mat', False), 16), (('tipu', False), 15), (('ampe', False), 15), (('ampit', False), 20), (('amput', False), 20), (('tempe', False), 15), (('empu', False), 10), (('top', False), 11), (('tut', False), 16), (('amat', False), 21), (('ampo', False), 15), (('taat', False), 15), (('tapi', False), 10), (('tepi', False), 10), (('emat', False), 15), (('tumpat', True), 24), (('umpat', True), 20), (('topi', False), 10), (('tempo', False), 15), (('tepat', False), 15), (('tampu', False), 15), (('tuat', False), 15), (('tempa', False), 15), (('tamat', False), 20), (('umat', False), 21), (('tempat', False), 20), (('tip', False), 16)]






    'kenapa awak tak suka makan Husein kat situ amat'




```python
normalizer.normalize(another,debug=True)
```

    [(('sika', False), 15), (('spa', False), 11), (('saka', False), 21), (('soka', False), 15), (('seka', False), 15), (('suka', False), 15), (('sua', False), 11), (('ski', False), 11), (('sia', False), 11)]

    [(('makna', False), 25), (('makin', False), 20), (('mani', False), 15), (('ikon', False), 15), (('kan', False), 16), (('ikan', False), 21), (('menu', False), 10), (('mini', False), 10), (('mena', False), 15), (('min', False), 11), (('makanan', False), 29), (('maun', False), 15), (('akan', False), 21), (('main', False), 15), (('makan', False), 25), (('kun', False), 11), (('kon', False), 11), (('ken', False), 11), (('mana', False), 21)]

    [(('kampung', True), 29)]

    [(('temut', False), 20), (('tempa', False), 20), (('tempat', False), 24), (('tempo', False), 20), (('tempe', False), 20)]






    'saya memang tak saka makanan Husein kampung tempat'




```python
normalizer = malaya.deep_normalizer()
```

    1.00MB [00:00, 2.66kMB/s]                  

    downloading JSON normalizer
    downloading normalizer graph



    22.0MB [00:05, 4.51MB/s]                          



```python
normalizer.normalize(string)
```




    'eye uau tak suka makan unsein kati situ tumpat'
