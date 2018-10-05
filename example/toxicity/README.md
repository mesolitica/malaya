

```python
import malaya
```

    Using TensorFlow backend.
    /usr/local/lib/python3.6/site-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
string = 'Benda yg SALAH ni, jgn lah didebatkan. Yg SALAH xkan jadi betul. Ingat tu. Mcm mana kesat sekalipun org sampaikan mesej, dan memang benda tu salah, diam je. Xyah nk tunjuk kau open sangat nk tegur cara org lain berdakwah. '
another_string = 'bodoh, dah la gay, sokong lgbt lagi, memang tak guna'
```


```python
model = malaya.multinomial_detect_toxic()
```


```python
model.predict(string)
```




    {'toxic': 0,
     'severe_toxic': 0,
     'obscene': 0,
     'threat': 0,
     'insult': 0,
     'identity_hate': 0}




```python
model.predict(string,get_proba=True)
```




    {'toxic': 0.14165235977019472,
     'severe_toxic': 1.9272487152616215e-06,
     'obscene': 0.011323038998473341,
     'threat': 8.249039905334012e-08,
     'insult': 0.008620760536227347,
     'identity_hate': 4.703244329372946e-06}




```python
model.predict(another_string)
```




    {'toxic': 1,
     'severe_toxic': 0,
     'obscene': 0,
     'threat': 0,
     'insult': 1,
     'identity_hate': 0}




```python
model.predict(another_string,get_proba=True)
```




    {'toxic': 0.97624511869432,
     'severe_toxic': 0.0004143925305717536,
     'obscene': 0.48936571876841484,
     'threat': 5.809081616106756e-06,
     'insult': 0.7853970362543069,
     'identity_hate': 0.002109806847753244}




```python
model.predict_batch([string,another_string])
```




    {'toxic': [0, 1],
     'severe_toxic': [0, 0],
     'obscene': [0, 0],
     'threat': [0, 0],
     'insult': [0, 1],
     'identity_hate': [0, 0]}




```python
model.predict_batch([string,another_string],get_proba=True)
```




    {'toxic': [0.14165235977019472, 0.97624511869432],
     'severe_toxic': [1.9272487152616215e-06, 0.0004143925305717536],
     'obscene': [0.011323038998473341, 0.48936571876841484],
     'threat': [8.249039905334012e-08, 5.809081616106756e-06],
     'insult': [0.008620760536227347, 0.7853970362543069],
     'identity_hate': [4.703244329372946e-06, 0.002109806847753244]}




```python
model = malaya.logistics_detect_toxic()
```


```python
model.predict(string)
```




    {'toxic': 0,
     'severe_toxic': 0,
     'obscene': 0,
     'threat': 0,
     'insult': 0,
     'identity_hate': 0}




```python
model.predict(another_string)
```




    {'toxic': 1,
     'severe_toxic': 0,
     'obscene': 0,
     'threat': 0,
     'insult': 0,
     'identity_hate': 0}




```python
model.predict_batch([string,another_string],get_proba=True)
```




    {'toxic': [0.10299208923447233, 0.6297643126911581],
     'severe_toxic': [0.010195223990855215, 0.019551370640497476],
     'obscene': [0.04834509566263489, 0.1995748012804703],
     'threat': [0.003488478318883341, 0.004014463652898358],
     'insult': [0.04528784776538583, 0.3354069432946268],
     'identity_hate': [0.011326619000125776, 0.052626041879065236]}




```python
model = malaya.deep_toxic()
```


```python
model.predict(string)
```




    {'toxic': 0.020363407,
     'severe_toxic': 0.0013132466,
     'obscene': 0.019614585,
     'threat': 0.0036143456,
     'insult': 0.017462607,
     'identity_hate': 0.008712418}




```python
model.predict(another_string)
```




    {'toxic': 0.4042334,
     'severe_toxic': 0.062299214,
     'obscene': 0.30533484,
     'threat': 0.031151816,
     'insult': 0.2637121,
     'identity_hate': 0.12611088}




```python
model.predict_batch([string,another_string])
```




    [{'toxic': 0.020363383,
      'severe_toxic': 0.0013132466,
      'obscene': 0.019614581,
      'threat': 0.0036143418,
      'insult': 0.017462632,
      'identity_hate': 0.008712414},
     {'toxic': 0.7303366,
      'severe_toxic': 0.024953173,
      'obscene': 0.32346514,
      'threat': 0.004960555,
      'insult': 0.3901479,
      'identity_hate': 0.053236082}]




```python

```
