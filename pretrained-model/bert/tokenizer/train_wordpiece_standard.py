from tokenizers import BertWordPieceTokenizer
from glob import glob

tokenizer = BertWordPieceTokenizer(
    clean_text=True, handle_chinese_chars=True, strip_accents=True, lowercase=False,
)
files = glob('splitted/*')
trainer = tokenizer.train(
    files,
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    limit_alphabet=1000,
    wordpieces_prefix="##",
)


tokenizer.save_model('./', 'bert-standard')
