## Generate preprocessing texts

1. Run all jupyter notebooks.

## Generating SentencePiece

```bash
spm_train \
--input=dumping-instagram.txt,dumping-twitter.txt,parliament-text.txt,wiki-text.txt,news-text.txt \
--model_prefix=sp10m.cased.v8 \
--vocab_size=30000 \
--character_coverage=0.99995 \
--model_type=unigram \
--control_symbols=\<cls\>,\<sep\>,\<pad\>,\<mask\>,\<eod\> \
--user_defined_symbols=\<eop\>,.,\(,\),\",-,–,£,€ \
--shuffle_input_sentence \
--input_sentence_size=10000000
```
