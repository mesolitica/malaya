## how-to

1. Download and preprocessing IIUM dataset, [preprocessing-iium.ipynb](preprocessing-iium.ipynb).

2. Download and preprocessing Instagram dataset, [preprocessing-instagram.ipynb](preprocessing-instagram.ipynb).

3. Download and preprocessing news dataset, [preprocessing-news.ipynb](preprocessing-news.ipynb).

4. Download and preprocessing parliament dataset, [preprocessing-parliament.ipynb](preprocessing-parliament.ipynb).

5. Download and preprocessing PDF dataset, [preprocessing-pdf.ipynb](preprocessing-pdf.ipynb).

6. Download and preprocessing Twitter dataset, [preprocessing-twitter.ipynb](preprocessing-twitter.ipynb).

7. Download and preprocessing wattpad dataset, [preprocessing-wattpad.ipynb](preprocessing-wattpad.ipynb).

8. Download and preprocessing Wikipedia dataset, [preprocessing-wiki.ipynb](preprocessing-wiki.ipynb).

9. Generate SentencePiece,

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