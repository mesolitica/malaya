from create_pretraining_data import process_documents
from glob import glob
import tokenization
from multiprocessing import Pool
import itertools
import os

tokenizer = tokenization.FullTokenizer(
    vocab_file = 'pegasus.wordpiece', do_lower_case = False
)

files = glob('/home/husein/pure-text/splitted/*.txt')

os.system('mkdir tfrecord')


def loop(files):
    for file in files:
        output_files = (
            f'tfrecord/pegasus-random-{os.path.split(file)[1]}.tfrecord'
        )
        process_documents(
            file, output_files, tokenizer, min_slide = 7, use_rouge = False
        )


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def multiprocessing(strings, function, cores = 10):
    df_split = chunks(strings, len(strings) // cores)
    pool = Pool(cores)
    pooled = pool.map(function, df_split)
    pool.close()
    pool.join()


multiprocessing(files, loop)
