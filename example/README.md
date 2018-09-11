# README-Examples

#### 1. Entities

Entities recognition, supported entities please check [here](entities/README.md).

to use multinomial model,
```python
malaya.multinomial_entities(string)
```

#### 2. Entities-POS

Entities and POS recognition, supported entities and POS please check [here](entities-pos/README.md).

to use deep learning model,
```python
malaya.deep_pos_entities(model).predict(string)
```

#### 3. Language detection

Language detection, able to detect {INDONESIA, ENGLISH, MALAY, OTHERS}.

classify single sentence,
```python
malaya.detect_language(text)
```

classify multiple sentences,
```python
malaya.detect_language(list_of_texts)
```

#### 4. Normalizer

Normalize words based on corpus given.

```python
corpus_normalize = ['maka','serious','yeke','masing-masing']
normalizer = malaya.naive_normalizer(corpus_normalize)
normalizer.normalize('masing2')
```

#### 5. Num2Word

number to words.

number to cardinal,
```python
malaya.to_cardinal(number)
```

number to ordinal,
```python
malaya.to_ordinal(number)
```

number to currency,
```python
malaya.to_currency(number)
```

number to year
```python
malaya.to_year(value)
```

#### 6. Sentiment

Classify negative and positive for documents. Please check it [here](sentiment/README.md).

#### 7. Stemmer

Stemming words.

```python
malaya.naive_stemmer(string)
```
