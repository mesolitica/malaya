# XLNET-Bahasa

Thanks to [zihangdai](https://github.com/zihangdai) for opensourcing XLNET, https://github.com/zihangdai/xlnet

## Objective

1. There is no multilanguage implementation of XLNET, and obviously no Bahasa Malaysia implemented. So this directory to provide pretraining XLNET for Bahasa Malaysia.

## How-to

1. Git clone [Malaya-Dataset](https://github.com/huseinzol05/Malaya-Dataset),

```bash
git clone https://github.com/huseinzol05/Malaya-Dataset.git
```

2. Run [tokenization.ipynb](tokenization.ipynb) to create dictionary for tokenizer and text dataset for pretraining.

3. Git clone [Sentence-Piece](https://github.com/google/sentencepiece),

```bash
git clone https://github.com/google/sentencepiece.git
```

4. Install [Sentence-Piece](https://github.com/google/sentencepiece),

On 23rd June 2019, I cannot use latest master to compile sentence-piece using bazel, after a few googled, we need to revert to some commit.

```bash
cd sentencepiece
git checkout d4dd947fe71c4fa4ee24ad8297beee32887d8828
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
```

Make sure you tested to run `spm_train` to make sure everything is fine,

```bash
spm_train
```

```text
ERROR: --input must not be empty

sentencepiece

Usage: sentencepiece [options] files

   --accept_language (comma-separated list of languages this model can accept)  type: string  default:
   --add_dummy_prefix (Add dummy whitespace at the beginning of text)  type: bool  default: true
   --bos_id (Override BOS (<s>) id. Set -1 to disable BOS.)  type: int32  default: 1
   --bos_piece (Override BOS (<s>) piece.)  type: string  default: <s>
   --character_coverage (character coverage to determine the minimum symbols)  type: double  default: 0.9995
   --control_symbols (comma separated list of control symbols)  type: string  default:
   --eos_id (Override EOS (</s>) id. Set -1 to disable EOS.)  type: int32  default: 2
...
```

5. Create tokenizer using Sentence-Piece,

```bash
cd ../
spm_train \
--input=texts.txt \
--model_prefix=sp10m.cased.v3 \
--vocab_size=32000 \
--character_coverage=0.99995 \
--model_type=unigram \
--control_symbols=\<cls\>,\<sep\>,\<pad\>,\<mask\>,\<eod\> \
--user_defined_symbols=\<eop\>,.,\(,\),\",-,–,£,€ \
--shuffle_input_sentence \
--input_sentence_size=10000000
```

**In the future, I will use Malaya tokenizer as XLNET tokenizer, if XLNET accuracies beat BERT accuracies.**

6. Convert text files to tfrecord,

```bash
mkdir save-location
python3 data_utils.py \
  --bsz_per_host=8 \
  --seq_len=256 \
  --reuse_len=128 \
  --input_glob=*.txt \
  --save_dir=save-location \
  --num_passes=20 \
  --bi_data=True \
  --sp_path=sp10m.cased.v3.model \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --num_core_per_host=1
```

7. Run pretained,

I reduce the size of XLNET by 2 while maintain the number of attention, here is [original size](https://github.com/zihangdai/xlnet#pretraining-with-xlnet).

```bash
python3 train_gpu.py \
  --corpus_info_path=save-location/corpus_info.json \
  --record_info_dir=save-location/tfrecords \
  --train_batch_size=8 \
  --seq_len=256 \
  --reuse_len=128 \
  --perm_size=128 \
  --n_layer=12 \
  --d_model=512 \
  --d_embed=512 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=2048 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85 \
  --model_dir=output-model \
  --uncased=True \
  --num_core_per_host=1
```
