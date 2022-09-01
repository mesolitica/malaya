## how-to

1. Download dataset at https://github.com/huseinzol05/malay-dataset/tree/master/dumping/clean
2. Generate SentencePiece,

```
LD_LIBRARY_PATH=/home/husein/lib
export LD_LIBRARY_PATH
spm_train \
--input=dumping-pdf.txt,dumping-news.txt,dumping-parliament.txt,dumping-wiki.txt \
--model_prefix=sp10m.cased.t5 \
--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 \
--vocab_size=32000 \
--character_coverage=0.99995 \
--model_type=unigram \
--user_defined_symbols=.,\(,\),\",-,–,£,€,\#,\' \
--shuffle_input_sentence \
--input_sentence_size=20000000
```

```
LD_LIBRARY_PATH=/home/husein/lib
export LD_LIBRARY_PATH
spm_train \
--input=dumping-pdf.txt,dumping-news.txt,dumping-parliament.txt,dumping-wiki.txt \
--model_prefix=sp10m.cased.t5-4k \
--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 \
--vocab_size=4000 \
--character_coverage=0.99995 \
--model_type=unigram \
--user_defined_symbols=.,\(,\),\",-,–,£,€,\#,\' \
--shuffle_input_sentence \
--input_sentence_size=20000000
```