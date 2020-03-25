## Generate preprocessing texts

1. Run all jupyter notebooks.

2. Generating SentencePiece for xlnet

```bash
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

3. Generating SentencePiece for bert and albert

```bash
spm_train \
--input=dumping-iium.txt,dumping-watpadd.txt,dumping-instagram.txt,dumping-news.txt,dumping-parliament.txt,dumping-pdf.txt,dumping-twitter.txt,dumping-wiki.txt \
--model_prefix=sp10m.cased.v10 \
--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 \
--vocab_size=32000 \
--character_coverage=0.99995 \
--model_type=unigram \
--control_symbols=\[CLS\],\[SEP\],\[MASK\] \
--user_defined_symbols=.,\(,\),\",-,–,£,€,\#,\',\[,\] \
--shuffle_input_sentence \
--input_sentence_size=20000000
```