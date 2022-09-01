## how-to

1. Download dataset at https://github.com/huseinzol05/malay-dataset/tree/master/dumping/clean
2. Generate SentencePiece,

```bash
LD_LIBRARY_PATH=/home/husein/lib
export LD_LIBRARY_PATH
spm_train \
--input=dumping-iium.txt,dumping-watpadd.txt,dumping-instagram.txt,dumping-news.txt,dumping-parliament.txt,dumping-pdf.txt,dumping-twitter.txt,dumping-wiki.txt \
--model_prefix=sp10m.cased.v9 \
--vocab_size=32000 \
--character_coverage=0.99995 \
--model_type=unigram \
--control_symbols=\<cls\>,\<sep\>,\<pad\>,\<mask\>,\<eod\> \
--user_defined_symbols=\<eop\>,.,\(,\),\",-,–,£,€,\#,\' \
--shuffle_input_sentence \
--input_sentence_size=20000000
```