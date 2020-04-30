from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=False, handle_chinese_chars=True, strip_accents=True, lowercase=False,
)


from glob import glob

files = glob('../bert/dumping-*.txt')
files = [i for i in files if 'twitter' not in i and 'instagram' not in i and 'combined' not in i] + ['dumping-commmon-crawl.txt']
files

trainer = tokenizer.train(
    files,
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)


tokenizer.save('./', 'bahasa-standard')