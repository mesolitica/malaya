## prepare sentencepiece

```bash
LD_LIBRARY_PATH=/home/husein/lib
export LD_LIBRARY_PATH
spm_train \
--input=bahasa/train-long-text/right.txt,bahasa/train-long-text/left.txt,bahasa/test-long-text/right.txt,bahasa/test-long-text/left.txt \
--model_prefix=sp10m.cased.translation \
--vocab_size=32000 \
--character_coverage=0.99995 \
--model_type=unigram \
--control_symbols=\<cls\>,\<sep\>,\<pad\>,\<mask\>,\<eod\> \
--user_defined_symbols=\<eop\>,.,\(,\),\",-,–,£,€,\#,\' \
--shuffle_input_sentence \
--input_sentence_size=20000000
```