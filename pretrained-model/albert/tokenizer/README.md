## how-to

1. Download dataset from https://github.com/huseinzol05/malay-dataset/tree/master/dumping/clean
2. Generate SentencePiece,

```bash
LD_LIBRARY_PATH=/home/husein/lib
export LD_LIBRARY_PATH
spm_train \
--input=dumping-iium.txt,dumping-watpadd.txt,dumping-news.txt,dumping-parliament.txt,dumping-pdf.txt,dumping-wiki.txt \
--model_prefix=sp10m.cased.albert \
--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
--vocab_size=32000 \
--character_coverage=0.99995 \
--model_type=unigram \
--control_symbols=\[CLS\],\[SEP\],\[MASK\] \
--user_defined_symbols=.,\(,\),\",-,–,£,€,\#,\',\[,\] \
--shuffle_input_sentence \
--input_sentence_size=20000000
```