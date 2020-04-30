from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

from glob import glob

files = glob('../bert/dumping-*.txt')
files = [
    i
    for i in files
    if 'twitter' not in i and 'instagram' not in i and 'combined' not in i
] + ['dumping-commmon-crawl.txt']
files

# same size as GPT 345M vocab, 'models/345M/hparams.json'
tokenizer.train(
    files,
    vocab_size = 50257,
    show_progress = True,
    special_tokens = ['<s>', '<pad>', '</s>', '<|endoftext|>'],
)


tokenizer.save('./', 'bahasa')
