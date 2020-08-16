## how-to

### bert-base

```
python3 src/main_bert_base.py train --use-bert --model-path-base models/en_bert --bert-model "bert-large-uncased" --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 --predict-tags --train-path train.txt --dev-path test.txt
```

### tiny-bert

```
python3 src/main_tiny_bert.py train --use-bert --model-path-base models/en_bert --bert-model "bert-large-uncased" --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 --predict-tags --train-path train.txt --dev-path test.txt
```

### albert-base

```
python3 src/main_albert_base.py train --use-bert --model-path-base models/en_bert --bert-model "bert-large-uncased" --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 --predict-tags --train-path train.txt --dev-path test.txt
```

### albert-tiny

```
python3 src/main_albert_tiny.py train --use-bert --model-path-base models/en_bert --bert-model "bert-large-uncased" --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 --predict-tags --train-path train.txt --dev-path test.txt
```

### xlnet-base

```
python3 src/main_xlnet_base.py train --use-bert --model-path-base models/en_bert --bert-model "bert-large-uncased" --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 --predict-tags --train-path train.txt --dev-path test.txt
```

### alxlnet-base

```
python3 src/main_alxlnet_base.py train --use-bert --model-path-base models/en_bert --bert-model "bert-large-uncased" --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 --predict-tags --train-path train.txt --dev-path test.txt
```