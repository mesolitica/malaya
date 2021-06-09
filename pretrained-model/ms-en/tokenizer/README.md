## how-to

1. download and prepare dataset , [download-translation-dataset.ipynb](download-translation-dataset.ipynb).

2. Prepare SentencePiece,

```
LD_LIBRARY_PATH=/home/husein/lib
export LD_LIBRARY_PATH
spm_train \
--input=source,target \
--model_prefix=sp10m.cased.ms-en \
--pad_id=0 --eos_id=1 --unk_id=2 --bos_id=-1 \
--vocab_size=32000 \
--character_coverage=0.99995 \
--model_type=unigram \
--user_defined_symbols=.,\(,\),\",-,–,£,€,\#,\' \
--shuffle_input_sentence \
--input_sentence_size=5000000
```
