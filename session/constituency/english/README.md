## how-to

1. git clone, https://github.com/michaeljohns2/self-attentive-parser

2. run training,

```bash
python3 src/main.py train --use-bert --model-path-base models/en_bert --bert-model "bert-large-uncased" --num-layers 2 --learning-rate 0.00005 --batch-size 32 --eval-batch-size 16 --subbatch-max-tokens 500 --predict-tags
```